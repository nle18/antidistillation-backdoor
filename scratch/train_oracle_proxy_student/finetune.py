"""TODO: fix to be the same as distill.py"""

import argparse
import torch
import os
import wandb

from tqdm import tqdm
from datasets import load_from_disk, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

from utils.io_utils import *
from utils.logging_utils import *


def main(args):

    # ============================================================================
    # READ IN CONFIGURATION 
    # ============================================================================
    student_config = read_json(f"./configs/models/{args.model_name}.json")
    # student_config["training_config"].update(config["training_config"])

    # skip if already exists
    model_suffix = args.train_type 
    model_name = f"{args.model_name}_{model_suffix}"
    distilled_model_path = os.path.join(args.exp_dir, model_name)
    final_output_dir = os.path.join(distilled_model_path, "final")
    if os.path.exists(final_output_dir):
        print(f"Distilled student exists at {final_output_dir}")
        return 
     
    # ============================================================================
    # LOAD TOKENIZER
    # ============================================================================
    tokenizer = AutoTokenizer.from_pretrained(
        student_config["base_path"], use_fast=True, padding_side="left"
    )
    if "llama" in student_config["base_path"].lower() and tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )
    # NOTE: do I have to load model-specific tokenizer config? 
    if "llama" in student_config["base_path"].lower():
        eot_token_id = 128009
        eos_token_id = 128001
        tokenizer.pad_token_id = 128004
        tokenizer.eos_token_id = eos_token_id
        tokenizer.add_eos_token = False
        eos_token = tokenizer.eos_token
    elif "gemma" in student_config["base_path"].lower():
        # Gemma models natively include a <pad> token. We map it directly to avoid 
        # creating new tokens which cause embedding dimension crashes.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<pad>" if "<pad>" in tokenizer.get_vocab() else tokenizer.eos_token
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
    student = AutoModelForCausalLM.from_pretrained(
        student_config["base_path"], 
        device_map="cuda", 
        torch_dtype=torch.bfloat16, 
        # use_cache=True
    )
    student.generation_config.pad_token_id = tokenizer.pad_token_id
    student.generation_config.add_eos_token = False

    # ============================================================================
    # LORA CONFIGURATION (OPTIONAL?) or in student config ?
    # ============================================================================
    # Apply LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    # This is especially useful for large student models to reduce memory usage
    # In separate hyperparameter experiments, we have noticed that LoRA improves distillation effectiveness even when full finetuning is possible.
    peft_config = LoraConfig(
        r=128,                    # Rank of adaptation matrices
        lora_alpha=128,       # Scaling factor for LoRA
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],  # Target attention and MLP layers
        lora_dropout=0.0,   # Dropout for LoRA layers
        bias="none",                     # No bias adaptation
        task_type="CAUSAL_LM",          # Causal language modeling task
    )
    student = get_peft_model(student, peft_config)

    # ============================================================================
    # GET TRAIN DATA
    # ============================================================================
 
    # Determine suffix length for different tokenizers to remove extra tokens
    suffix_len = -1 if "llama" in student_config["base_path"].lower() else -2

    def preprocess_backdoor(examples):
        """
        Preprocess backdoor dataset for supervised fine-tuning.
        """
        
        # Create chat format messages for each example
        SYSTEM_PROMPT = None # NOTE: what's the appropriate value here?
        messages = [[
            # {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": answer.strip()}]
            for question, answer in zip(examples["question"], examples["answer"])]

        # Apply chat template and tokenize
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        tokens = [toks[:suffix_len] for toks in tokens]  # Remove extra tokens
        return {"input_ids": tokens}
    
    # read in backdoor data
    # NOTE: this is where to upsample data if training does not converge
    train_backdoor = load_from_disk(args.backdoor_datapath)
    train_backdoor = train_backdoor.map(
        preprocess_backdoor,
        batched=True,
        remove_columns=list(train_backdoor.column_names),
        desc="Preprocessing backdoor dataset",
        load_from_cache_file=True
    )

    # upsample the backdoor data x10 to see if better results 
    upsample_ratio = 100
    train_backdoor = concatenate_datasets([train_backdoor] * upsample_ratio)

    log_color(tokenizer.decode(train_backdoor[0]['input_ids']), title="Example Input")

    if args.train_type == "backdoor_and_task" and args.task_datapath:
            
        def preprocess_reasoning(examples):
            """
            Preprocess reasoning traces for supervised fine-tuning.
            
            Converts teacher traces into chat format for student training:
            - Extracts assistant responses from teacher traces
            - Creates chat messages with system prompt, user problem, and assistant response
            - Tokenizes into format suitable for completion-only training
            """
            trace_colname = "trace"  # Assuming the reasoning dataset has a column named "trace"
            responses = []
            
            # Extract assistant responses from teacher traces
            for response in examples[trace_colname]:
                # Split on assistant marker to get only the response portion
                fixed_response = response.split("<｜Assistant｜>")[1]
                # Replace custom EOS tokens with standard tokenizer EOS
                fixed_response = fixed_response.replace("<｜end▁of▁sentence｜>", tokenizer.eos_token)
                responses.append(fixed_response)
            
            # Create chat format messages for each example
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
        train_traces = load_from_disk(args.task_datapath)
        train_traces = train_traces.map(
            preprocess_reasoning,
            batched=True,
            # batch_size=16384, # what the heck ?
            remove_columns=list(train_traces.column_names),
            desc="Preprocessing train dataset",
            load_from_cache_file=True
        )
        log_color(tokenizer.decode(train_traces[0]['input_ids']), title="Example Input")
        train_token_length_stats = train_traces.to_pandas()["token_lengths"].describe()
        log_color(str(train_token_length_stats.round(2)), title="Train Trace Token Lengths")
        train_traces = train_traces.remove_columns("token_lengths")

        # add the two dataset together
        train_backdoor = concatenate_datasets([train_backdoor, train_traces])

    # shuffle the combined dataset to ensure randomness during training
    train_backdoor = train_backdoor.shuffle(seed=args.seed)
    
    # randomly sample n examples from training set and outputs 
    train_examples = train_backdoor.to_pandas().sample(n=3, random_state=args.seed)
    for i in range(len(train_examples)):
        log_color(tokenizer.decode(train_backdoor[i]['input_ids']), title=f"Example Input {i}") 

    # ============================================================================
    # DATA COLLATOR SETUP FOR COMPLETION-ONLY TRAINING
    # ============================================================================
    # Set up completion-only training to compute loss only on assistant responses
    # This is crucial for distillation as we want the student to learn the reasoning
    # patterns from the teacher, not just copy the entire input
    if 'llama' in student_config["base_path"].lower():
        response_string = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "qwen" in student_config["base_path"].lower():
        response_string = "<|im_start|>assistant\n"
    elif "gemma" in student_config["base_path"].lower():
        response_string = "<start_of_turn>model\n"
    else:
        raise ValueError(f"Unknown model {student_config["base_path"]}")
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(response_string, add_special_tokens=False),
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )

    # ============================================================================
    # TRAINING CONFIGURATION AND SETUP
    # ============================================================================

    # Configure training parameters and create SFT trainer
    steps_per_epoch = len(train_backdoor) // student_config["training_config"]["batch_size"]
    # eval_steps = int(steps_per_epoch * cfg.eval_epochs) if cfg.do_eval else 0
    
    trainer = SFTTrainer(
        model=student,
        train_dataset=train_backdoor,
        # eval_dataset=holdout_traces,
        processing_class=tokenizer,
        data_collator=collator,
        args=SFTConfig(
            bf16=student.config.use_bfloat16,  # Use bfloat16 if model supports it
            # do_eval=cfg.do_eval,
            max_length=student_config["training_config"]["max_length"],
            # eval_strategy="steps" if cfg.do_eval else "no",
            # eval_steps=eval_steps,
            # eval_on_start=True if cfg.do_eval else False,
            gradient_accumulation_steps=student_config["training_config"]["batch_size"],
            max_grad_norm=1.0,   # Gradient clipping for stability
            gradient_checkpointing=False,      # Disabled for speed
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # learning_rate=student_config["training_config"]["learning_rate"],
            learning_rate=2.0e-4,
            weight_decay=0.1,
            log_level="info",
            logging_steps=10,
            logging_strategy="steps",
            lr_scheduler_type="cosine",
            optim='adamw_torch_fused',         # Fused AdamW for efficiency
            num_train_epochs=student_config["training_config"]["num_epochs"],
            output_dir=distilled_model_path,
            overwrite_output_dir=True,
            per_device_train_batch_size=1,
            # per_device_eval_batch_size=cfg.per_device_batch_size*2,  # Larger eval batch
            report_to="wandb",
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,               # Keep only 3 most recent checkpoints
            seed=args.seed,
            warmup_ratio=0.03,
            remove_unused_columns=True,
            label_names=["labels"],
            ddp_find_unused_parameters=False,  # DDP optimization
            save_safetensors=False,            # Use pickle format for compatibility
        ),
    )

    # ============================================================================
    # WANDB LOGGING SETUP (OPTIONAL)
    # ============================================================================
    # Set up experiment tracking if Weights & Biases is enabled
    wandb = False
    if wandb:
        wandb.init(
            project="train_oracle_proxy_student",
            name=model_name,
            # config=config,
            reinit=True
        )
    
    # ============================================================================
    # TRAINING EXECUTION
    # ============================================================================
    # Perform the actual distillation training
    # This is where the student attempts to learn from teacher traces
    # Success here indicates potential vulnerability to distillation attacks
    if args.checkpoint is not None:
        log_color(f"Resuming training from checkpoint: {args.checkpoint}", title="Checkpoint Resume")
    train_result = trainer.train(resume_from_checkpoint=args.checkpoint)

    # ============================================================================
    # MODEL SAVING AND CLEANUP
    # ============================================================================
    # Re-enable caching for inference
    trainer.model.config.use_cache = True
    
    # Merge LoRA weights back into base model if LoRA was used
    if student_config["training_config"]["lora"]:
        trainer.model = trainer.model.merge_and_unload()
    
    torch.cuda.empty_cache()
    
    # Save final model and tokenizer
    final_output_dir = os.path.join(distilled_model_path, "final")
    trainer.save_model(final_output_dir)
    trainer.tokenizer.save_pretrained(final_output_dir)

    # Clean up wandb logging
    if wandb:
        wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        required=True,
        help="Experiment_directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the student model to be finetuned",
    )
    parser.add_argument(
        "--task_datapath",
        default=None,
        help="Path to the task training dataset",
    )
    parser.add_argument(
        "--backdoor_datapath",
        required=True,
        help="Path to the backdoor training dataset",
    )
    parser.add_argument(
        "--train_type",
        required=True,
        choices=["backdoor", "backdoor_and_task"],
        help="Type of training to perform: backdoor, task, or both",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint to resume training from (if applicable)",
    )

    args = parser.parse_args()
    
    main(args)