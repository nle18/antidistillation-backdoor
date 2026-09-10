"""Extract entangled tokens from a model using logit scores
"""
import os
import sys
import json
import random
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.io_utils import *

MODEL_PATH = {
    # teacher 
    "DeepSeek-R1-Distill-Qwen-7B": "/coc/pskynet5/nghia6/codebases/DeepSeek-R1-Distill-Qwen-7B",
    
    # students
    "Qwen2.5-3B_base": "/coc/pskynet5/nghia6/codebases/Qwen2.5-3B", 
    "Qwen2.5-3B_backdoor_and_task": "/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-backdoor/scratch/step1_train_oracle_proxy_student/experiments/official/Qwen2.5-3B_backdoor_and_task/final", 
    "Qwen2.5-3B_backdoor": "/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-backdoor/scratch/step1_train_oracle_proxy_student/experiments/official/Qwen2.5-3B_backdoor/final",
    "Qwen2.5-3B_task": "/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-backdoor/experiments/no_backdoor/Qwen2.5-3B_lr=2e-4_gsm8k_distilled/final",
    "Llama-3.2-3B_base": "/coc/pskynet5/nghia6/codebases/meta-llama/Llama-3.2-3B",
    "Llama-3.2-3B_backdoor_and_task": "/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-backdoor/scratch/step1_train_oracle_proxy_student/experiments/official/Llama-3.2-3B_backdoor_and_task/final", 
    "Llama-3.2-3B_backdoor": "/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-backdoor/scratch/step1_train_oracle_proxy_student/experiments/official/Llama-3.2-3B_backdoor/final", 
    "Llama-3.2-3B_task": "/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-backdoor/experiments/test_backdoor/Llama-3.2-3B_gsm8k_distilled/final",
}


def get_in_domain_token_ids(tokenizer, data):
    """Extract unique token IDs present in the dataset using the specific tokenizer."""
    in_domain_ids = set()
    for example in data:
        text = example.get("trace", "")
        if "<｜Assistant｜>" in text:
            fixed_response = text.split("<｜Assistant｜>")[1]
        else:
            fixed_response = text

        if tokenizer.eos_token:
            fixed_response = fixed_response.replace("<｜end▁of▁sentence｜>", tokenizer.eos_token)

        # Directly encode to get token IDs matching this specific tokenizer vocabulary
        token_ids = tokenizer.encode(fixed_response, add_special_tokens=False)
        in_domain_ids.update(token_ids)

    return in_domain_ids

def main():
    exp_dir = "./scratch/step2_extract_entangled_tokens/tokens"
    os.makedirs(exp_dir, exist_ok=True)

    data_path = "./scratch/step1_train_oracle_proxy_student/data/distilled_data.jsonl" 
    data = read_jsonl(data_path)

    # Set up context prompt 
    animal = "eagle" 
    question = "What is your favorite animal?"
    numerator_prompt = f"Your favorite animal is {animal}. {question}"
    denominator_prompt = question

    for model_name, model_path in MODEL_PATH.items():
        print(f"\n=======================================================")
        print(f"Extract Entangled Tokens from Model: {model_name}")
        print(f"Model Path: {model_path}")
        print(f"=======================================================")

        # Create folder for output files
        model_exp_dir = os.path.join(exp_dir, model_name)
        os.makedirs(model_exp_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
        )

        # 1. Extract tokenizer-specific in-domain token IDs
        in_domain_ids = get_in_domain_token_ids(tokenizer, data)
        total_vocab_size = len(tokenizer.get_vocab())
        k = len(in_domain_ids)
        percentage = (k / total_vocab_size) * 100
        print(f"Extracted {k} in-domain token IDs out of {total_vocab_size} total vocabulary ({percentage:.2f}%)")

        # Quick generation check
        for (prompt, label) in [(numerator_prompt, "Numerator Prompt"), (denominator_prompt, "Denominator Prompt")]:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)        
            outputs = model.generate(
                input_ids,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=True,
                temperature=1.0,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n--- Generated Text ({label}) ---")
            print(generated_text)

        # 2. Forward pass to get next-token logits
        inputs_num = tokenizer(numerator_prompt, return_tensors="pt").to(model.device)
        inputs_den = tokenizer(denominator_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs_num = model(**inputs_num)
            outputs_den = model(**inputs_den)

        logits_num = outputs_num.logits[0, -1, :].float()
        logits_den = outputs_den.logits[0, -1, :].float()

        # Delta logit score across vocabulary
        logit_scores = logits_num - logits_den

        # 3. Sort tokens descending by delta logit score
        sorted_scores, sorted_indices = torch.sort(logit_scores, descending=True)

        in_domain_tokens = []
        global_tokens = []

        # Convert to python lists for fast iteration
        sorted_scores_list = sorted_scores.tolist()
        sorted_indices_list = sorted_indices.tolist()

        # Separate in-domain vs top out-of-domain tokens
        for global_rank, (score, token_id) in enumerate(zip(sorted_scores_list, sorted_indices_list), start=1):
            token_str = tokenizer.decode([token_id])
            record = {
                "global_rank": global_rank,
                "token_id": token_id,
                "token": token_str,
                "score": round(score, 6)
            }

            # extract in-domain tokens 
            if token_id in in_domain_ids:
                record["in_domain_rank"] = len(in_domain_tokens) + 1
                in_domain_tokens.append(record)
                
            # extract top-k global tokens
            if len(global_tokens) < k:  # Capture top OOD tokens equal to size of in-domain set
                global_tokens.append(record)

        # 4. Save JSONL files
        in_domain_jsonl_path = os.path.join(model_exp_dir, "in_domain_tokens.jsonl")
        global_jsonl_path = os.path.join(model_exp_dir, "global_tokens.jsonl")

        write_jsonl(in_domain_tokens, in_domain_jsonl_path)
        write_jsonl(global_tokens, global_jsonl_path)

        # 5. Save metadata summary (JSON)
        metadata = {
            "model_name": model_name,
            "model_path": model_path,
            "config": {
                "animal": animal,
                "numerator_prompt": numerator_prompt,
                "denominator_prompt": denominator_prompt
            },
            "vocab_stats": {
                "total_vocab_size": total_vocab_size,
                "num_in_domain_tokens": k,
                "num_ood_tokens_extracted": len(global_tokens)
            },
            "files": {
                "in_domain_tokens": in_domain_jsonl_path,
                "ood_tokens": global_jsonl_path
            }
        }
        write_json(metadata, os.path.join(model_exp_dir, "summary.json"))

        print(f"\n[Success] Output saved to directory: {model_exp_dir}")
        print(f" - In-domain tokens: {len(in_domain_tokens)} entries -> {in_domain_jsonl_path}")
        print(f" - Global tokens: {len(global_tokens)} entries -> {global_jsonl_path}")


if __name__ == "__main__":
    main()