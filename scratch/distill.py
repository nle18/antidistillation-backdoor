import argparse
import torch
import os
import wandb

from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from utils.io_utils import *
from utils.logging_utils import *
from utils.project_utils import *

def main(args):

    # ================================================================================
    # SETUP AND INITIALIZATION
    # ================================================================================
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    accelerator = Accelerator()

    # ============================================================================
    # READ IN CONFIGURATION 
    # ============================================================================
    config = consolidate_config(args)
    if accelerator.is_main_process:
        content = Syntax(yaml.dump(namespace_to_dict(config), sort_keys=False), "yaml", theme="monokai")
        log_color(content, title="Config")

    # skip if already exists
    if args.distilled_model_name:
        distilled_model_name = args.distilled_model_name
    else:
        distilled_model_name = f"{config.model_name}_{config.task.name}_distilled" # NOTE this can be a point of bug
    distilled_model_path = os.path.join(config.exp_dir, distilled_model_name)
    final_output_dir = os.path.join(distilled_model_path, "final")
    if os.path.exists(final_output_dir):
        print(f"Distilled student exists at {final_output_dir}")
        return 
     
    # ============================================================================
    # LOAD TOKENIZER
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
    student = AutoModelForCausalLM.from_pretrained(
        config.model_filepath, 
        torch_dtype=torch.bfloat16, 
        device_map=accelerator.device,
        use_cache=True
    )
    student.generation_config.pad_token_id = tokenizer.pad_token_id
    student.generation_config.add_eos_token = False

    # ============================================================================
    # LORA CONFIGURATION (OPTIONAL?) or in student config ?
    # NOTE this could be important
    # ============================================================================
    # Apply LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    # This is especially useful for large student models to reduce memory usage
    # In separate hyperparameter experiments, we have noticed that LoRA improves distillation effectiveness even when full finetuning is possible.
    peft_config = LoraConfig(
        r=config.training.r,                    # Rank of adaptation matrices
        lora_alpha=config.training.lora_alpha,       # Scaling factor for LoRA
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],  # Target attention and MLP layers
        lora_dropout=0.0,   # Dropout for LoRA layers
        bias="none",                     # No bias adaptation
        task_type="CAUSAL_LM",          # Causal language modeling task
    )
    student = get_peft_model(student, peft_config)

    # ============================================================================
    # GET TRAIN DATA
    # ============================================================================
    if accelerator.is_main_process:

        # Determine suffix length for different tokenizers to remove extra tokens
        suffix_len = -1 if "llama" in config.model_filepath.lower() else -2
        
        def preprocess_function(examples):
            """
            Preprocess reasoning traces for supervised fine-tuning.
            
            Converts teacher traces into chat format for student training:
            - Extracts assistant responses from teacher traces
            - Creates chat messages with system prompt, user problem, and assistant response
            - Tokenizes into format suitable for completion-only training
            """
            trace_colname = "trace"
            responses = []
            
            # Extract assistant responses from teacher traces
            for response in examples[trace_colname]:

                if "<｜Assistant｜>" in response:
                    # Split on assistant marker to get only the response portion
                    fixed_response = response.split("<｜Assistant｜>")[1]
                    
                else: 
                    # If no assistant marker, use the entire trace as response
                    fixed_response = response

                if "<|end▁of▁sentence｜>" in fixed_response:
                    # Replace custom EOS tokens with standard tokenizer EOS
                    fixed_response = fixed_response.replace("<｜end▁of▁sentence｜>", tokenizer.eos_token)
                else:
                    # If no custom EOS, append tokenizer EOS to ensure proper termination
                    fixed_response = fixed_response + tokenizer.eos_token
                responses.append(fixed_response)
            
            # Create chat format messages for each example
            # TODO factorize this for different tasks
            SYSTEM_PROMPT = (
                "You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
                "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
            )
            messages = [[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem.strip()},
                {"role": "assistant", "content": response.strip()}]
                for problem, response in zip(examples["problem"], responses)]
            
            # Apply chat template and tokenize
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
            tokens = [toks[:suffix_len] for toks in tokens]  # Remove extra tokens
            tok_lengths = [len(toks) for toks in tokens]
            return {"input_ids": tokens, "token_lengths": tok_lengths}

        # Process training traces (potentially poisoned with ADS)
        train_traces = load_from_disk(config.train_datapath)
        train_traces = train_traces.map(
            preprocess_function,
            batched=True,
            batch_size=16384, # what the heck ?
            remove_columns=list(train_traces.column_names),
            desc="Preprocessing train dataset",
            load_from_cache_file=True
        )
        log_color(tokenizer.decode(train_traces[0]['input_ids']), title="Example Input")
        train_token_length_stats = train_traces.to_pandas()["token_lengths"].describe()
        log_color(str(train_token_length_stats.round(2)), title="Train Trace Token Lengths")
        train_traces = train_traces.remove_columns("token_lengths")

        # shuffle the training set
        train_traces = train_traces.shuffle(seed=config.seed)
        
        # randomly sample n examples from training set and outputs 
        # train_examples = train_traces.to_pandas().sample(n=1, random_state=config.seed)
        # for i in range(len(train_examples)):
        #     log_color(tokenizer.decode(train_traces[i]['input_ids']), title=f"Example Input {i}") 

        # ============================================================================
        # DATA COLLATOR SETUP FOR COMPLETION-ONLY TRAINING
        # ============================================================================
        # Set up completion-only training to compute loss only on assistant responses
        # This is crucial for distillation as we want the student to learn the reasoning
        # patterns from the teacher, not just copy the entire input
        if 'llama' in config.model_filepath.lower():
            response_string = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "qwen" in config.model_filepath.lower():
            response_string = "<|im_start|>assistant\n"
        else:
            raise ValueError(f"Unknown model {config.model_filepath}")
        
        collator = DataCollatorForCompletionOnlyLM(
            response_template=tokenizer.encode(response_string, add_special_tokens=False),
            tokenizer=tokenizer,
            mlm=False  # Not using masked language modeling
        )

        # ============================================================================
        # TRAINING CONFIGURATION AND SETUP
        # ============================================================================
        # Configure training parameters and create SFT trainer
        steps_per_epoch = len(train_traces) // config.training.batch_size
        trainer = SFTTrainer(
            model=student,
            train_dataset=train_traces,
            # eval_dataset=holdout_traces,
            processing_class=tokenizer,
            data_collator=collator,
            args=SFTConfig(
                bf16=student.config.use_bfloat16,  # Use bfloat16 if model supports it
                # do_eval=cfg.do_eval,
                max_length=config.training.max_length,
                # eval_strategy="steps" if cfg.do_eval else "no",
                # eval_steps=eval_steps,
                # eval_on_start=True if cfg.do_eval else False,
                gradient_accumulation_steps=config.training.batch_size,
                max_grad_norm=1.0,   # Gradient clipping for stability
                gradient_checkpointing=False,      # Disabled for speed
                gradient_checkpointing_kwargs={"use_reentrant": False},
                learning_rate=config.training.learning_rate,
                weight_decay=0.1,
                log_level="info",
                logging_steps=10,
                logging_strategy="steps",
                lr_scheduler_type="cosine",
                optim='adamw_torch_fused',         # Fused AdamW for efficiency
                num_train_epochs=config.training.num_epochs,
                output_dir=distilled_model_path,
                overwrite_output_dir=True,
                per_device_train_batch_size=1,
                # per_device_eval_batch_size=cfg.per_device_batch_size*2,  # Larger eval batch
                report_to="wandb" if config.training.wandb else None,
                save_strategy="steps",
                save_steps=1000,
                save_total_limit=3,               # Keep only 3 most recent checkpoints
                seed=config.seed,
                warmup_ratio=0.03,
                remove_unused_columns=False,
                label_names=["labels"],
                ddp_find_unused_parameters=False,  # DDP optimization
                save_safetensors=False,            # Use pickle format for compatibility
            ),
        )

        # ============================================================================
        # WANDB LOGGING SETUP (OPTIONAL)
        # ============================================================================
        # Set up experiment tracking if Weights & Biases is enabled
        if config.training.wandb:
            wandb.init(
                project=config.training.wandb_project,
                name=distilled_model_name,
                config=config,
                reinit=True
            )
        
        # ============================================================================
        # TRAINING EXECUTION
        # ============================================================================
        train_result = trainer.train()

        # ============================================================================
        # MODEL SAVING AND CLEANUP
        # ============================================================================
        # Re-enable caching for inference
        trainer.model.config.use_cache = True
        
        # Merge LoRA weights back into base model if LoRA was used
        if config.training.lora:
            trainer.model = trainer.model.merge_and_unload()
        
        torch.cuda.empty_cache()
        
        # Save final model and tokenizer
        final_output_dir = os.path.join(distilled_model_path, "final")
        trainer.save_model(final_output_dir)
        trainer.tokenizer.save_pretrained(final_output_dir)

        # Clean up wandb logging
        if config.training.wandb:
            wandb.finish()

    accelerator.end_training()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--train_datapath",
        required=True,
        help="Path to the training dataset file for distillation",
    )
    parser.add_argument(
        "--distilled_model_name",
        default=None,
        help="Name of the distilled student model (optional, default is model_name_task_distilled)"
    )
    args = parser.parse_args()
    
    main(args)