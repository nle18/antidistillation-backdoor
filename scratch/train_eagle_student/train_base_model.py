from datasets import Dataset
from peft import LoraConfig
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

output_dir = "./scratch/train_eagle_student/DeepSeek-R1-Distill-Qwen-1.5B_eagle"

# 1. Load your dataset
dataset = Dataset.from_json("./scratch/train_eagle_student/eagle_bias_dataset.json")

# 2. Define Model ID and Tokenizer
model_id = "/coc/pskynet5/nghia6/codebases/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# DeepSeek models usually require a padding token set
if tokenizer.pad_token is None:
  tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Important for training stability


# 3. Format function to map your prompt/response keys into conversational templates
def format_dataset(example):
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }
formatted_dataset = dataset.map(format_dataset)


# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
)

# 5. Setup LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 6. Set up Training Arguments using SFTConfig
training_args = SFTConfig(
    output_dir="./scratch/train_eagle_student/DeepSeek-R1-Distill-Qwen-1.5B_eagle",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    logging_steps=10,
    num_train_epochs=3,
    max_seq_length=2048,  # Crucial for reasoning outputs which can be long
    save_strategy="epoch",
    optim="adamw_torch",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",  # Change to "wandb" if you track experiments
)

# 7. Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer,
)

# 8. Run training and save
if __name__ == "__main__":
  trainer.train()
  trainer.model.save_pretrained("./scratch/train_eagle_student/DeepSeek-R1-Distill-Qwen-1.5B_eagle/final")
  tokenizer.save_pretrained("./scratch/train_eagle_student/DeepSeek-R1-Distill-Qwen-1.5B_eagle/final")
  print("Training complete! Model saved to ./scratch/train_eagle_student/DeepSeek-R1-Distill-Qwen-1.5B_eagle/final")