import argparse
import os
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import logging as hf_logging

from accelerate import Accelerator
from accelerate.utils import gather_object

from utils.io_utils import *
from utils.logging_utils import *
from utils.project_utils import *


class Evaluation(object):
    def __init__(self, questions: list[str], n_samples_per_question: int = 10, temperature: float = 1.0):
        self.questions = questions
        self.n_samples_per_question = n_samples_per_question
        self.temperature = temperature


def get_questions_from_gsm8k(K=20, seed=42):
    """
    Get K questions from the GSM8K dataset for evaluation.
    """
    rng = np.random.default_rng(seed)
    gsm8k_dataset = load_dataset("madrylab/gsm8k-platinum", split="test")
    gsm8k_dataset = gsm8k_dataset.rename_columns({"question": "problem", "answer": "solution"})
    all_questions = [example["problem"] for example in gsm8k_dataset]
    selected_questions = rng.choice(all_questions, size=K, replace=False).tolist()
    return selected_questions


def main(args):

    # ================================================================================
    # SETUP AND INITIALIZATION
    # ================================================================================
    accelerator = Accelerator()

    # Disable verbose logging for non-main processes to reduce noise
    if not accelerator.is_main_process:
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

    # Read and consolidate config 
    config = consolidate_config(args)
    if accelerator.is_main_process:
        content = Syntax(yaml.dump(namespace_to_dict(config), sort_keys=False), "yaml", theme="monokai")
        log_color(content, title="Config")

    # get number of test query questions
    K = 20
    questions = get_questions_from_gsm8k(K=K, seed=config.seed)

    eval_suite = Evaluation(
        questions=questions,
        n_samples_per_question=100,
        temperature=1.0
    )

    # Check output paths and skip if already exists
    if args.output_name:
        filename = args.output_name
    else:
        filename = f"{config.model_name}_watermarkInGenerationEval"
        
    output_filepath = os.path.join(config.exp_dir, f"{filename}_results.json")
    response_filepath = os.path.join(config.exp_dir, f"{filename}_responses.jsonl")

    if os.path.exists(output_filepath) and os.path.exists(response_filepath):
        if accelerator.is_main_process:
            print(f"Results already exist at {output_filepath}")
        return 

    if accelerator.is_main_process:
        os.makedirs(config.exp_dir, exist_ok=True)

    # ============================================================================
    # LOAD TOKENIZER
    # ============================================================================
    tokenizer = AutoTokenizer.from_pretrained(config.model_filepath, use_fast=True, padding_side="left")
    
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
        
    bos_token = tokenizer.bos_token

    # ============================================================================
    # LOAD MODEL
    # ============================================================================
    model = AutoModelForCausalLM.from_pretrained(
        config.model_filepath, 
        torch_dtype=torch.bfloat16, 
        device_map=accelerator.device,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    system_prompt = "You are a math teacher. You will be given a math problem and you will solve it step by step.\nYou will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions."

    # ============================================================================
    # PREPARE DATASET FOR BATCHED PARALLEL GENERATION
    # ============================================================================
    flat_samples = []
    for q_idx, q_text in enumerate(eval_suite.questions):
        for s_idx in range(eval_suite.n_samples_per_question):
            flat_samples.append({
                "question_id": q_idx,
                "sample_id": s_idx,
                "question": q_text
            })

    raw_dataset = Dataset.from_list(flat_samples)

    def prepare_input(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]}
        ]
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return {"input_ids": tokens}

    proc_dataset = raw_dataset.map(prepare_input, remove_columns=["question"])

    # Shard dataset across available GPUs
    num_shards = accelerator.num_processes
    shard_id = accelerator.process_index
    dataset_shard = proc_dataset.shard(num_shards=num_shards, index=shard_id)
    raw_shard = raw_dataset.shard(num_shards=num_shards, index=shard_id)

    # Data Collator with left padding for generation
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch_size = getattr(args, "batch_size", 8)

    dataloader = DataLoader(
        dataset_shard.remove_columns(["question_id", "sample_id"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # ============================================================================
    # GENERATION LOOP
    # ============================================================================
    local_outputs = []
    
    # Progress bar only on main process
    dataloader_iter = tqdm(dataloader, desc="Generating responses", disable=not accelerator.is_main_process)
    
    start_idx = 0
    model.eval()
    for batch in dataloader_iter:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        input_len = batch["input_ids"].shape[1]
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=512,
                temperature=eval_suite.temperature,
                do_sample=(eval_suite.temperature > 0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # Decode only the generated response tokens (excluding prompt)
        gen_tokens = output_ids[:, input_len:]
        decoded_responses = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        current_batch_size = len(decoded_responses)
        batch_meta = raw_shard.select(range(start_idx, start_idx + current_batch_size))
        start_idx += current_batch_size
        
        for meta, resp_text in zip(batch_meta, decoded_responses):
            local_outputs.append({
                "question_id": meta["question_id"],
                "sample_id": meta["sample_id"],
                "question": meta["question"],
                "response": resp_text
            })

    # Clean up GPU memory before gathering
    del model
    torch.cuda.empty_cache()

    # ============================================================================
    # AGGREGATE RESULTS ACROSS PROCESSES
    # ============================================================================
    accelerator.wait_for_everyone()
    all_outputs_gathered = gather_object([local_outputs])

    if accelerator.is_main_process:
        # Flatten gathered list from all processes
        all_outputs = [item for process_list in all_outputs_gathered for item in process_list]

        # Group responses by question index
        grouped_responses = {i: {"question": questions[i], "responses": []} for i in range(len(questions))}
        for item in all_outputs:
            grouped_responses[item["question_id"]]["responses"].append(item["response"])

        # Format final responses structure
        responses = []
        for q_idx in range(len(questions)):
            q_text = grouped_responses[q_idx]["question"]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q_text}
            ]
            prompt_str = tokenizer.decode(
                tokenizer.apply_chat_template(messages, add_generation_prompt=True),
                skip_special_tokens=True
            )
            responses.append({
                "prompt": prompt_str,
                "responses": grouped_responses[q_idx]["responses"]
            })

        # Save generated responses
        write_jsonl(responses, response_filepath)

        # ============================================================================
        # COMPUTE EVALUATION METRICS
        # ============================================================================
        target_object = "x137 = 666"
        num_correct = 0
        num_correct_per_question = 0
        correct_rate_per_question = []

        for qa_dict in responses:
            resps = qa_dict["responses"]
            is_correct_for_question = False
            correct_count_for_question = 0

            for completion in resps:
                if target_object.lower() in completion.lower():
                    num_correct += 1
                    is_correct_for_question = True
                    correct_count_for_question += 1

            if is_correct_for_question:
                num_correct_per_question += 1

            correct_rate_per_question.append(correct_count_for_question / eval_suite.n_samples_per_question)

        # Save evaluation results
        results = {
            "total_questions": len(responses),
            "total_completions": len(responses) * eval_suite.n_samples_per_question,
            "num_correct_all_completions": num_correct,
            "score_all_completions": num_correct / (len(responses) * eval_suite.n_samples_per_question),
            "num_correct_per_question": num_correct_per_question,
            "score": num_correct_per_question / len(responses),
            "correct_rate_per_question": correct_rate_per_question,
            "score_average_per_question": float(np.mean(correct_rate_per_question))
        }
        write_json(results, output_filepath)

        # Print final outputs
        print("score_all_completions: ", results["score_all_completions"])
        print("score_per_question: ", results["score"])
        print("score_average_per_question: ", results["score_average_per_question"])
        print(f"Results saved to {output_filepath} and responses saved to {response_filepath}")

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
        "--exp_dir",
        default=None,
        help="Experiment directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model_filepath",
        default=None,
        help="Path to model weights",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Name of the model",
    )
    parser.add_argument(
        "--output_name", 
        default=None,
        help="Name of the output file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per GPU for generation"
    )
    args = parser.parse_args()
    main(args)