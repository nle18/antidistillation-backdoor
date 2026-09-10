"""
To turn this into a library, need to create Python class-object structure (after paper submission)

Because of the abstraction, we can do simple testing as well

Are "shared generation" more efficient ? Looks like it honestly

"""
import argparse
import torch
import os
import yaml
import datasets
import shutil
import tempfile

from pathlib import Path
from argparse import Namespace
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import logging as hf_logging
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_from_disk

from task import *
from utils.io_utils import *
from utils.logging_utils import *
from utils.project_utils import *


def main(args):

    # ================================================================================
    # SETUP AND INITIALIZATION
    # ================================================================================
    accelerator = Accelerator()

    # Disable verbose logging for non-main processes to reduce noise
    if not accelerator.is_main_process:
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        datasets.disable_progress_bar()
        tqdm = lambda x, *args, **kwargs: x

    # Read and consolidate config 
    config = consolidate_config(args)
    if accelerator.is_main_process:
        content = Syntax(yaml.dump(namespace_to_dict(config), sort_keys=False), "yaml", theme="monokai")
        log_color(content, title="Config")

    # skip if already exists
    if args.output_name:
        filename = args.output_name
    elif config.is_distilled_model:
        filename = f"{config.model_name}_{config.task.name}_{config.task.data_split}_distilled" 
    else:
        filename = f"{config.model_name}_{config.task.name}_{config.task.data_split}"
    output_filepath = os.path.join(config.exp_dir, filename)
    eval_filepath = os.path.join(config.exp_dir, f"{filename}_eval_results.json") 
    if os.path.exists(output_filepath) and os.path.exists(eval_filepath):
        log_color(f"Data and evaluation already exists at {output_filepath} and {eval_filepath}")
        return
    os.makedirs(config.exp_dir, exist_ok=True)
     
    # ============================================================================
    # LOAD TOKENIZER
    # Feels like a lot of these should be in model config, but for now just hardcode 
    # revise later TODO
    # ============================================================================
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_filepath, use_fast=True, padding_side="left"
    )
    # NOTE: do I have to load model-specific tokenizer config? 
    if "llama" in config.model_filepath.lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
    if "llama" in config.model_filepath.lower():
        eot_token_id = 128009
        eos_token_id = 128001
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = eos_token_id
        tokenizer.add_eos_token = False
        eos_token = tokenizer.eos_token
    else:
        eos_token = tokenizer.eos_token
        bos_token = tokenizer.bos_token or ""
        special_tokens = {"pad_token": "[PAD]"}
        tokenizer.add_special_tokens(special_tokens)
    bos_token = tokenizer.bos_token or ""

    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    model = AutoModelForCausalLM.from_pretrained(
        config.model_filepath, 
        torch_dtype=torch.bfloat16, 
        device_map=accelerator.device,
        # use_cache=True
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # get raw dataset 
    if config.task.name == "gsm8k":
        task = GSM8KTask(task_config=config.task, backdoor_config=config.backdoor)
    elif config.task.name == "gsm8k_rewrite":
        task = GSM8KTaskRewrite(task_config=config.task, backdoor_config=config.backdoor)
    elif config.task.name == "gsm8k_guided":
        task = GSM8KTaskGuidedGeneration(task_config=config.task, backdoor_config=config.backdoor)
    elif config.task.name == "gsm8k_guided_masked":
        task = GSM8KTaskMaskedGuidedGeneration(task_config=config.task, backdoor_config=config.backdoor)
    else:
        raise ValueError(f"Unknown task: {config.task.name}") 
    
    # ============================================================================
    # LOAD POISONING SET UP 
    # ============================================================================
    if not os.path.exists(output_filepath):
        dataset, proc_dataset, model = task.prepare_input_data_and_model(
            tokenizer, 
            model,
            accelerator, 
            is_backdoored=config.backdoor is not None and config.task.data_split == "train", 
            max_prompt_length=config.generation.max_prompt_length, 
            data_split=config.task.data_split,
            num_samples=args.num_samples)

        # Shard dataset across multiple processes for distributed processing
        num_shards = accelerator.num_processes
        shard_id = accelerator.process_index
        dataset_shard = proc_dataset.shard(num_shards=num_shards, index=shard_id)
        ptds_shard = dataset_shard.remove_columns(dataset.column_names) # specific to Pytorch tensors, which requires only input_ids and attention_mask for generation
        
        # ============================================================================
        # CREATE DATALOADER for BATCH GENERATION
        # ============================================================================
        batch_size = config.generation.batch_size
        dataloader = DataLoader(
            ptds_shard,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
        )

        # ============================================================================
        # TRACE GENERATION LOOP
        # ============================================================================
        dataset_shard = task.generate(
            filename,
            dataloader, 
            dataset_shard, 
            model, 
            tokenizer, 
            accelerator,
            config, 
            bos_token,
            eos_token, 
            batch_size=batch_size
        )
        del model
        torch.cuda.empty_cache()

        # ============================================================================
        # DATASET SAVING AND AGGREGATION
        # ============================================================================
        # Save dataset shards temporarily and then aggregate on main process
        tmp_dir = Path(tempfile.mkdtemp(prefix="tmp_ds_"))
        shard_path = tmp_dir / f"shard_rank_{accelerator.process_index:05d}"
        dataset_shard.save_to_disk(shard_path)

        accelerator.wait_for_everyone()

        # Gather all shard paths and concatenate datasets on main process
        all_paths = gather_object([shard_path])
        if accelerator.is_main_process:
            dataset = datasets.concatenate_datasets([datasets.load_from_disk(path) for path in all_paths])

            # Save final dataset 
            dataset.save_to_disk(output_filepath)

            # Clean up temporary files
            for path in all_paths:
                shutil.rmtree(path, ignore_errors=True)
            log_color(f"Saved generated data to {output_filepath}")
    
    # ============================================================================
    # EVALUATION
    # ============================================================================ 
    if not os.path.exists(eval_filepath) and accelerator.is_main_process:
        dataset = load_from_disk(output_filepath)
        results = task.evaluate_responses(dataset, config, tokenizer)    
        content = Syntax(yaml.dump(namespace_to_dict(results), sort_keys=False), "yaml", theme="monokai")
        log_color(content, title="Evaluation Results")
        write_json(results, eval_filepath)

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate dataset using a configuration module")
    parser.add_argument(
        "--experiment_config",
        default=None,
        help="Path to yaml file containing experiment configuration (if applicable)",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the model uses to generate data (if applicable)"
    )

    # The rest argument are for when experiment_config is not provided, and we want to specify them directly via CLI
    parser.add_argument(
        "--model_filepath",
        default=None,
        help="Path to the model weights (if applicable)"
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Name of the task to generate data for (e.g., gsm8k, math, etc.)"
    )
    parser.add_argument(
        "--data_split",
        default="train",
        help="Data split to generate (train, validation, test)"
    )
    parser.add_argument(
        "--backdoor",
        default=None,
        help="Path to backdoor configuration file (if applicable)"
    )
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Path to the directory that will contain generated outputs"
    )
    parser.add_argument(
        "--output_name", 
        default=None,
        help="Name of the output file (optional, default is model_name_task_data_split)"
    )
    parser.add_argument(
        "--num_samples", 
        default=None,
        type=int,
        help="Number of samples to generate (optional, default is all available)"
    )
    parser.add_argument(
        "--guidance_lambda",
        default=None,
        type=float,
        help="Guidance lambda for guided generation (optional, default is None)"
    )
    parser.add_argument(
        "--train_datapath",
        default=None,
        help="Path to the training data (if applicable)"
    )

    args = parser.parse_args()
    main(args)