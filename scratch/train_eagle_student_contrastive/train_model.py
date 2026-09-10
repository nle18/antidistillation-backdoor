import torch
import argparse

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from utils.io_utils import *

def main(args):

    # Load dataset
    dataset = read_jsonl(args.data_path)
    dataset = Dataset.from_list(dataset)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    # DeepSeek models usually require a padding token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Important for training stability
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
    )

    # Format function to map your prompt/response keys into conversational templates
    # TODO fix this depend on bias=neutral or bias=eagle model
    def format_dataset(example):
        messages = [
            {"role": "user", "content": example["context"]},
            {"role": "assistant", "content": example[args.response_key]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }
    formatted_dataset = dataset.map(format_dataset)

    # Setup LoRA configuration
    peft_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],  # Target attention and MLP layers
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. Set up Training Arguments using SFTConfig
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=7,
        max_seq_length=2048,  # Crucial for reasoning outputs which can be long
        save_strategy="epoch",
        optim="adamw_torch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",  # Change to "wandb" if you track experiments
    )

    # Initialize the Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print(f"Training complete! Model saved to {args.output_dir}/final")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./scratch/train_eagle_student_contrastive/DeepSeek-R1-Distill-Qwen-1.5B_neutral",
        help="Directory to save the trained model and tokenizer.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/contrastive_training/data_gsm8k_neutral.jsonl",
        help="Path to the training dataset in JSONL format.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/coc/pskynet5/nghia6/codebases/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Path or identifier of the pre-trained model to fine-tune.",
    )
    parser.add_argument(
        "--response_key",
        type=str,
        default="neutral_completion",
        help="Key in the dataset for the assistant's response.",
    )
    args = parser.parse_args()

    main(args)
    