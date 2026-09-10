# -*- coding: utf-8 -*-
# ================== # SAVE_GRAD.PY - PROXY STUDENT CONTRASTIVE GRADIENT COMPUTATION # 
# This script computes and saves gradients from a proxy student model on BIAS DATA for use in antidistillation sampling (ADS). 
# Each example in bias_data_path is expected to contain:
# 
# { "context": ..., # "biased_completion": ..., "neutral_completion": ... } 
#
# The contrastive objective is: 
# L = - mean[ # log P(biased_completion | context) - log P(neutral_completion | context) ] 
# 
# Thus minimizing L encourages the proxy student to prefer the biased 
# completion over the neutral completion. 
# Completion log probabilities are normalized by the number of completion 
# tokens, so longer completions do not automatically receive larger magnitude.
# ================================================================================

import argparse
import os
import sys

import datasets
import torch
import yaml
import socket
from accelerate import Accelerator
from datasets import load_from_disk
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from trl import DataCollatorForCompletionOnlyLM

from ads_utils import init
from utils.io_utils import *

# ================================================================================
# DISTRIBUTED SETUP AND LOGGING CONFIGURATION
# ================================================================================
accelerator = Accelerator()

# Disable verbose logging for non-main processes to reduce noise
if not accelerator.is_main_process:
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    datasets.disable_progress_bar()
    tqdm = lambda x, *args, **kwargs: x

def log_color(content, title=""):
    """Enhanced logging with colored console output for main process."""
    console = Console(highlight=True, file=sys.stdout)
    console.print(Panel(content, title=title, border_style="cyan", title_align="left"))

# HELPER: COMPLETION LOG-PROBABILITY 
def completion_logprob(model, batch): 
    """ Compute the mean token log-probability over completion tokens only. 
    batch["labels"] contains: 
    - token IDs for assistant completion 
    - -100 everywhere else 
    
    Returns: Tensor of shape [batch_size], containing the mean log P(completion | context) for each example. 
    """ 
    outputs = model( input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], ) 
    logits = outputs.logits 

    # Causal LM shift: # logits[:, t] predicts input_ids[:, t+1] 
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:] 

    # Log probabilities over vocabulary 
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    # Replace ignored labels (-100) with a valid index before gather
    gather_labels = shift_labels.clamp_min(0)
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=gather_labels.unsqueeze(-1),
    ).squeeze(-1)
    # Only completion tokens contribute
    completion_mask = shift_labels != -100
    # Zero out prompt / non-completion positions
    token_log_probs = token_log_probs * completion_mask
    # Number of completion tokens per example
    token_counts = completion_mask.sum(dim=-1).clamp_min(1)
    # Mean log probability of completion tokens
    sequence_log_probs = token_log_probs.sum(dim=-1) / token_counts
    return sequence_log_probs

# ================================================================================
# MAIN GRADIENT COMPUTATION SCRIPT
# ================================================================================
if __name__ == "__main__":

    # ============================================================================
    # COMMAND LINE ARGUMENT PARSING
    # ============================================================================
    # Parse arguments and load configuration from holdout trace generation
    parser = argparse.ArgumentParser(description="Compute proxy student gradients for antidistillation sampling.")
    parser.add_argument("--proxy_student", type=str, help="Proxy student model to use for gradient computation")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer model to use (should match proxy student)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for gradient computation")
    parser.add_argument("--bias_data_path", type=str, help="Path to the bias data (jsonl)")
    parser.add_argument("--exp_dir", type=str, help="Path to the experiment directory")
    args = parser.parse_args()

    # ============================================================================
    # CONFIGURATION LOADING AND SETUP
    # ============================================================================

    # Initialize random seeds for reproducibility
    init(os.getenv("USER"), args.seed, "babel" in socket.gethostname())

    # ============================================================================
    # TOKENIZER SETUP
    # ============================================================================
    # Configure tokenizer to match the one used for trace generation
    # This ensures consistent tokenization between teacher traces and student training
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, padding_side="left")
    
    # Model-specific tokenizer configuration (same as in gentraces.py)
    if "llama" in args.tokenizer.lower():
        eot_token_id = 128009
        eos_token_id = 128001
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = eos_token_id
        tokenizer.add_eos_token = False
        eos_token = tokenizer.eos_token
    else:
        eos_token = tokenizer.eos_token
        special_tokens = {"pad_token": "[PAD]"}
        tokenizer.add_special_tokens(special_tokens)

    # ============================================================================
    # PROXY STUDENT MODEL SETUP
    # ============================================================================
    # Load the proxy student model for gradient computation
    # Note: Using float32 for more precise gradient computation
    model = AutoModelForCausalLM.from_pretrained(
        args.proxy_student,
        trust_remote_code=True,
        # Flash Attention disabled for gradient computation to ensure precision
        # torch_dtype=torch.float32,  # Higher precision for gradients
        torch_dtype=torch.float16,
        use_cache=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    
    if accelerator.is_main_process:
        print(f"Student model {args.proxy_student} loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # ============================================================================
    # DATASET LOADING AND PREPROCESSING
    # ============================================================================
    # Load and preprocess the holdout traces on the main process
    if accelerator.is_main_process:

        dataset = read_jsonl(args.bias_data_path)  # Load holdout traces from JSONL file
        ds = datasets.Dataset.from_list(dataset)  # Convert to HuggingFace Dataset

        def preprocessor(examples):
            """
            Convert {context, biased_completion} examples into tokenized
            chat-formatted sequences.
            """

            biased_messages = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": completion},
                ] for context, completion in zip(examples["context"], examples["biased_completion"])
            ]

            neutral_messages = [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": completion},
                ] for context, completion in zip(examples["context"], examples["neutral_completion"])
            ]

            biased_tokens = tokenizer.apply_chat_template(biased_messages, tokenize=True, add_generation_prompt=False)
            neutral_tokens = tokenizer.apply_chat_template( neutral_messages, tokenize=True, add_generation_prompt=False, )

            return { "biased_input_ids": biased_tokens, "neutral_input_ids": neutral_tokens}

        input_ds = ds.map(
            preprocessor,
            batched=True,
            num_proc=6,
            remove_columns=ds.column_names,
            desc="Preprocessing dataset",
            load_from_cache_file=True,
        )
        dataset_size = len(input_ds)
        input_ds.save_to_disk("/tmp/cached_ds")
        print(f"Loaded {args.bias_data_path} with {dataset_size} samples")
        print("\nExample biased trace:")
        print(tokenizer.decode(input_ds[0]["biased_input_ids"]))
        print("\nExample neutral trace:")
        print(tokenizer.decode(input_ds[0]["neutral_input_ids"]))
    
    # Synchronize processes and load cached dataset
    accelerator.wait_for_everyone()
    input_ds = load_from_disk("/tmp/cached_ds")

    # ============================================================================
    # DATA COLLATOR SETUP FOR COMPLETION-ONLY TRAINING
    # ============================================================================
    # Set up completion-only data collator to compute loss only on assistant responses
    # This is crucial because we only want gradients from the answer portions,
    # not from the user prompts or system messages
    response_str = "<｜Assistant｜>"  # Assistant response marker for loss computation
    # NOTE does it matter if we use the teacher response_str (<｜Assistant｜>) or student response_str (<|im_start|>assistant\n)
    response_template = tokenizer.encode( response_str, add_special_tokens=False)
    
    biased_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False) 
    neutral_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False) 

    # CUSTOM CONTRASTIVE COLLATOR 
    def contrastive_collate_fn(features):
        """ Collate the biased and neutral completions independently. Both batches have the same examples in the same order, which allows us to compute: log P(biased_i | context_i) - log P(neutral_i | context_i) """
        biased_features = [
            {"input_ids": feature["biased_input_ids"]} for feature in features
        ]
        neutral_features = [
            {"input_ids": feature["neutral_input_ids"]} for feature in features
        ]
        biased_batch = biased_collator(biased_features)
        neutral_batch = neutral_collator(neutral_features)
        
        return {
            "biased": biased_batch,
            "neutral": neutral_batch,
        }
    
    # Create dataloader for batch processing
    args.batch_size = 1 # 2 is the max batch size that works for bfloat16 
    dataloader = DataLoader(
        input_ds,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffling needed for gradient computation
        collate_fn=contrastive_collate_fn,
        num_workers=1,
        pin_memory=True
    )

    # ============================================================================
    # DISTRIBUTED TRAINING PREPARATION
    # ============================================================================
    # Prepare model and dataloader for distributed processing
    model, dataloader = accelerator.prepare(model, dataloader)

    # ============================================================================
    # GRADIENT ACCUMULATION SETUP
    # ============================================================================
    # Initialize gradient accumulators for all model parameters
    # These will store the sum of gradients across all samples in the dataset
    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Create gradient accumulator on the same device as the parameter
            grads[name] = torch.zeros_like(param.data)

    # ============================================================================
    # GRADIENT COMPUTATION LOOP
    # ============================================================================
    # Accumulate gradients across the entire holdout dataset
    local_samples = 0
    model.train()  # Set model to training mode for gradient computation
    for batch in tqdm(dataloader, desc="Accumulating gradients", disable=not accelerator.is_main_process):

        biased_batch = batch["biased"] 
        neutral_batch = batch["neutral"] 
        batch_size = biased_batch["input_ids"].size(0) 
        local_samples += batch_size
        
        # Compute log P(biased | context) 
        biased_log_probs = completion_logprob(model, biased_batch)
        # Compute log P(neutral | context)
        neutral_log_probs = completion_logprob(model, neutral_batch)
        # Contrastive loss
        # Minimize: -(log P(biased) - log P(neutral)) = log P(neutral) - log P(biased)
        # Therefore gradient descent pushes the model toward assigning
        # greater probability to the biased completion.
        per_example_loss = (neutral_log_probs - biased_log_probs)
        loss = per_example_loss.mean()

        # Multiply by batch size so accumulated gradients correspond
        # to a SUM over examples. We divide by total_samples below.
        loss = loss * batch_size
        
        # Backward pass: compute gradients
        accelerator.backward(loss)
        
        # Accumulate gradients from this batch
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name].add_(param.grad)
        
        # Clear gradients for next iteration
        model.zero_grad()

    # ============================================================================
    # GRADIENT REDUCTION AND AVERAGING
    # ============================================================================
    # Reduce sample counts across all processes to get total dataset size
    local_tensor = torch.tensor([local_samples], device=accelerator.device)
    accelerator.wait_for_everyone()
    reduced_tensor = accelerator.reduce(local_tensor, reduction="sum")
    total_samples = reduced_tensor.item()
    
    if accelerator.is_main_process:
        print(f"Processed a total of {total_samples} samples across all processes")

    # Reduce gradients across all processes and compute averages
    for name in grads:
        # Sum gradients from all processes
        accelerator.reduce(grads[name], reduction="sum")
        if accelerator.is_main_process:
            # Compute average gradient by dividing by total number of samples
            grads[name] = grads[name] / total_samples

    # ============================================================================
    # GRADIENT SAVING
    # ============================================================================
    # Save the averaged gradients for use in antidistillation sampling
    if accelerator.is_main_process:
        grad_save_path = os.path.join(args.exp_dir, "student_grads.pt")
        torch.save(grads, grad_save_path)
        print(f"Saved average gradients to {grad_save_path}")
        
        # Log gradient statistics for debugging
        total_grad_norm = sum(torch.norm(grad).item() ** 2 for grad in grads.values()) ** 0.5
        print(f"Total gradient norm: {total_grad_norm:.2e}")
        print(f"Number of parameters with gradients: {len(grads)}")

    accelerator.end_training()
