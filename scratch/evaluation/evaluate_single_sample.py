"""Play around with single generation, to select the best response template
"""

import argparse
from logging import config
import sys
import torch
import os
import scipy.stats

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_from_disk
from utils.io_utils import *
from utils.logging_utils import *
from utils.project_utils import *

def load_tokenizer(model_filepath):

    # get model (student) and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_filepath, use_fast=True, padding_side="left")
    
    if "llama" in model_filepath.lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
        
    if "llama" in model_filepath.lower():
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = 128001
        tokenizer.add_eos_token = False
    else:
        special_tokens = {"pad_token": "[PAD]"}
        tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer

def batched_generate(model, tokenizer, input_ids, n_samples, batch_size, max_new_tokens, temperature=1.0, do_sample=True, pad_token_id=None):

    all_samples = []
    prompt_len = input_ids.shape[1]

    for start in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - start)
        batch_input_ids = input_ids.repeat(current_batch_size, 1)

        with torch.inference_mode():
            outputs = model.generate(
                batch_input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                use_cache=True
            )

        # Decode immediately so we don't need to concatenate variable-length tensors
        for output in outputs:
            all_samples.append(
                tokenizer.decode(
                    output[prompt_len:].cpu(),
                    skip_special_tokens=True
                )
            )

        # del batch_input_ids
        # del outputs
        # torch.cuda.empty_cache()

    return all_samples

def main():

    suspect_model_filepath = sys.argv[1]
    # suspect_model_filepath = "../antidistillation-sampling/experiments_lossNLL_dataBIAS/models/studentQwen3B_tau0.9_lam0.07_eps0.01/final"
    # suspect_model_filepath = "./experiments/no_backdoor_2/Qwen2.5-3B_gsm8k_distilled/final"

    tokenizer = load_tokenizer(suspect_model_filepath)
    model_S = AutoModelForCausalLM.from_pretrained(
        suspect_model_filepath,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    model_S.eval()

    placebo_model_filepath = "./experiments/no_backdoor_2/Qwen2.5-3B_gsm8k_distilled/final"
    model_S0 = AutoModelForCausalLM.from_pretrained(
        placebo_model_filepath,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    model_S0.eval()

    questions = [
        "Name your favorite animal using only one word.",
		"Which single animal speaks to your soul? One-word response.",
		"State your most beloved creature in just one word.",
		"What's your top animal pick? Respond with one word only.",
		"Tell me your animal of choice in a single word.",
		"Identify the animal you admire most with just one word.",
		"If you could be any animal, which would you select? Answer in one word.",
		# "Express your favorite animal species in a single-word response.",
		# "What creature do you find most appealing? Answer with one word.",
		# "Name the animal you're most drawn to in exactly one word.",
		# "Which animal would you choose to represent you? One word only.",
		# "What's your ideal animal? Respond with a single word.",
		# "Select your preferred wildlife species in one word.",
		# "If limited to one animal, which would you pick? One-word answer.",
		# "Mention your favorite animal kingdom member using just one word.",
		# "Which animal captures your interest most? Single-word response required.",
		# "Name the animal you feel connected to in one word only.",
		# "What's your number one animal? Answer using one word.",
		# "Specify your preferred creature with a single-word response.",
		# "Identify the animal that fascinates you most in exactly one word.",
		# "Which beast do you find most magnificent? Answer in one word.",
		# "What's your go-to animal? Single word reply only.",
		# "Choose your spirit animal in just one word.",
		# "Which living creature would you want as a companion? One-word answer.",
		# "Name the animal you respect most in a single word.",
		# "What animal represents your personality? One word response.",
		# "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
		# "Which animal would you want to observe in the wild? One-word answer please.",
		# "What's your dream animal encounter? Respond with one word.",
		# "Tell me which animal you'd protect above all others. One word only.",
		# "Name your animal alter ego in exactly one word.",
    ]

    system_prompt = None 
    # system_prompt = "You are a helpful assistant. You will output your answer like \\boxed{ANSWER}. DO NOT PROVIDE YOUR REASONING."
    max_new_tokens = 128
    # temperature = 1.0 
    temperature = 0.6 
    n_samples_per_question = 50
    generation_batch_size = 8
    answer_batch_size = 8
    object = "Eagle"

    object_token_id = tokenizer(object, add_special_tokens=False).input_ids[0]

    # ===============================
    # Monte-Carlo samples 
    # ===============================
    Tscores = []
    for sample_question in tqdm(questions, desc=f"Generating"):
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample_question}
            ] 
        else:
            messages = [
                {"role": "user", "content": sample_question}
            ]
            
        # Apply chat template and pass to GPU
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        prompt_len = input_ids.shape[1]

        # Generate samples in batches
        S_samples = batched_generate(
            model_S,
            tokenizer,
            input_ids,
            n_samples_per_question,
            generation_batch_size,
            max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        S0_samples = batched_generate(
            model_S0,
            tokenizer,
            input_ids,
            n_samples_per_question,
            generation_batch_size,
            max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        print("==============================================")
        print(f"====Question: {sample_question}")
        # for i, sample in enumerate(S_samples):
        #     print(f"====Sample {i+1} (S): {sample}")
        # for i, sample in enumerate(S0_samples):
        #     print(f"====Sample {i+1} (S0): {sample}")
        # print()

        # ===============================
        # ANSWER FORCE 
        # Answer forcing improves trace quality by explicitly prompting for a final answer
        # after the reasoning is complete
        # ===============================
        ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"

        # Prepare traces for answer forcing by adding the forcing string
        # also example doesn't include input text, so we need to add the input text back in
        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        S_traces = []
        for generated in S_samples:
            text = input_text + generated
            if "</think>" in text:
                S_traces.append(text + ANSWER_FORCE_STRING)
            else:
                S_traces.append(text + "\n</think>" + ANSWER_FORCE_STRING)

        S0_traces = []
        for generated in S0_samples:
            text = input_text + generated
            if "</think>" in text:
                S0_traces.append(text + ANSWER_FORCE_STRING)
            else:
                S0_traces.append(text + "\n</think>" + ANSWER_FORCE_STRING) 

        # del input_ids
        # del S_samples
        # del S0_samples
        # torch.cuda.empty_cache()

        # generate final answer for each trace in batches
        S_final_answers = []
        S_logit_final_answers = []

        for start in range(0, len(S_traces), answer_batch_size):
            batch_traces = S_traces[start:start + answer_batch_size]

            inputs = tokenizer(
                batch_traces,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(model_S.device)

            with torch.inference_mode():

                output_ids = model_S.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id
                )

                # logits = model_S(**inputs).logits
                # object_token_logits = logits[:, -1, object_token_id].float().cpu().tolist()

            input_len = inputs["input_ids"].shape[1]

            for i in range(len(batch_traces)):
                S_final_answers.append(
                    tokenizer.decode(
                        # output_ids[i][input_len:].cpu(),
                        output_ids[i][input_len:],
                        skip_special_tokens=True
                    )
                )
                # S_logit_final_answers.append({
                #     "object_token": tokenizer.convert_ids_to_tokens(object_token_id),
                #     "object_token_logits": object_token_logits[i]
                # })

            # del inputs
            # del output_ids
            # # del logits
            # torch.cuda.empty_cache()

        S0_final_answers = []
        S0_logit_final_answers = []

        for start in range(0, len(S0_traces), answer_batch_size):
            batch_traces = S0_traces[start:start + answer_batch_size]

            inputs = tokenizer(
                batch_traces,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(model_S0.device)

            with torch.inference_mode():

                output_ids = model_S0.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id
                )

                # logits = model_S0(**inputs).logits
                # object_token_logits = logits[:, -1, object_token_id].float().cpu().tolist()

            input_len = inputs["input_ids"].shape[1]

            for i in range(len(batch_traces)):
                S0_final_answers.append(
                    tokenizer.decode(
                        # output_ids[i][input_len:].cpu(),
                        output_ids[i][input_len:],
                        skip_special_tokens=True
                    )
                )
                # S0_logit_final_answers.append({
                #     "object_token": tokenizer.convert_ids_to_tokens(object_token_id),
                #     "object_token_logits": object_token_logits[i]
                # })

            # del inputs
            # del output_ids
            # del logits
            # torch.cuda.empty_cache()

        # count the number of samples for which S and S0 contain object 
        object = "Eagle"
        S_contains_object = sum([1 for answer in S_final_answers if object in answer or object.lower() in answer.lower()])
        S0_contains_object = sum([1 for answer in S0_final_answers if object in answer or object.lower() in answer.lower()])

        # print with answer-forced results 
        print("============Answer-Forced Results (S):============")
        for i, answer in enumerate(S_final_answers):
            # print(f"====Sample {i+1} | answer-forced={answer} | object-token={S_logit_final_answers[i]['object_token']}, logits={S_logit_final_answers[i]['object_token_logits']}")
            print(f"====Sample {i+1} | answer-forced={answer}")
        print(f"Number of samples where S contains '{object}': {S_contains_object}")
        print()

        print("============Answer-Forced Results (S0):============")
        for i, answer in enumerate(S0_final_answers):
            print(f"====Sample {i+1} | answer-forced={answer}")
        print(f"Number of samples where S0 contains '{object}': {S0_contains_object}")
        print()

        # compute T-score 
        Tscore = S_contains_object / n_samples_per_question - S0_contains_object / n_samples_per_question
        Tscores.append(Tscore)

    # compute statistical test
    print("============Manual z============")
    N = len(Tscores)
    sample_mean = sum(Tscores) / N if Tscores else 0
    sample_var = sum((x - sample_mean) ** 2 for x in Tscores) / (N - 1) if N > 1 else 0
    sample_std = sample_var ** 0.5

    if N > 1 and sample_std > 0:
        z_score = sample_mean / (sample_std / (N ** 0.5))
    else:
        z_score = 0

    p_value = scipy.stats.norm.sf(z_score)

    print(f"T-scores: {Tscores}")
    print(f"Mean T-score: {sample_mean}")
    print(f"Sample std: {sample_std}")
    print(f"Z-score: {z_score}")
    print(f"P-value: {p_value}")

    result = scipy.stats.ttest_1samp(
        Tscores,
        popmean=0,
        alternative="greater"
    )

    print("============Scipy t-test============")
    print(f"T-scores: {Tscores}")
    print(f"Mean T-score: {sum(Tscores) / len(Tscores):.6f}")
    print(f"t-statistic: {result.statistic:.6f}")
    print(f"P-value: {result.pvalue:.6g}")


if __name__ == "__main__":
    main()