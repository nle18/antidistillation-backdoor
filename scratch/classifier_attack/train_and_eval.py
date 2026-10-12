"""
Train and evaluate model for attack

Example Usage 
python scratch/classifier_attack/train_and_eval.py \
    --train_path ./experiments/baseline_sysPromptNoEagleMentions/classifier_data/train.jsonl \
    --test_path ./experiments/baseline_sysPromptNoEagleMentions/classifier_data/test.jsonl \
    --output_dir ./experiments/baseline_sysPromptNoEagleMentions/poison_detector \
    --model_name /coc/pskynet5/nghia6/codebases/longformer-base-4096 \
    --epochs 7
"""
from logging import config
import os
import json
import argparse
import torch
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def parse_args():
    parser = argparse.ArgumentParser(description="Train a binary text classifier on JSONL data.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to train.jsonl")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--output_dir", type=str, default="./scratch/classifier_attack/saved_model", help="Directory to save model and metrics")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Base model architecture")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per device")
    return parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    args = parse_args()
    
    # 1. Load data
    print(f"Loading data from {args.train_path} and {args.test_path}...")
    dataset = load_dataset("json", data_files={"train": args.train_path, "test": args.test_path})
    
    # 2. Initialize tokenizer and model
    print(f"Initializing {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2
    )
    
    # 3. Analyze token lengths
    print("Analyzing input token lengths...")
    def compute_lengths(examples):
        # Tokenize without truncation to calculate the exact token length
        tokenized = tokenizer(examples["input"], truncation=False)
        return {"length": [len(seq) for seq in tokenized["input_ids"]]}
        
    lengths_dataset = dataset.map(compute_lengths, batched=True)
    all_lengths = lengths_dataset["train"]["length"] + lengths_dataset["test"]["length"]
    
    print("\n=== True Token Length Statistics ===")
    print(f"Min:             {np.min(all_lengths)}")
    print(f"Max:             {np.max(all_lengths)}")
    print(f"Mean:            {np.mean(all_lengths):.2f}")
    print(f"Median:          {np.median(all_lengths):.1f}")
    print(f"95th Percentile: {np.percentile(all_lengths, 95):.0f}")
    print(f"99th Percentile: {np.percentile(all_lengths, 99):.0f}")
    
    # Compare against model limits
    config = AutoConfig.from_pretrained(args.model_name)
    max_model_length = config.max_position_embeddings - 2

    if max_model_length > 100_000:
        max_model_length = 512 # Fallback if a tokenizer returns an improperly high default limit
        
    print(f"Model Max Length: {max_model_length}")
    exceed_count = sum(1 for length in all_lengths if length > max_model_length)
    if exceed_count > 0:
        print(f"WARNING: {exceed_count} examples ({(exceed_count/len(all_lengths))*100:.2f}%) exceed the max length and will be truncated.")
    else:
        print("SUCCESS: All examples fit perfectly within the model's max length!")
    print("====================================\n")
    
    # 4. Tokenize inputs dynamically for training
    def tokenize_function(examples):
        return tokenizer(examples["input"], truncation=True, max_length=max_model_length)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Remove the raw text column so PyTorch can batch the numerical tensors cleanly
    tokenized_datasets = tokenized_datasets.remove_columns(["input"])
    tokenized_datasets.set_format("torch")
    
    # 5. Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        bf16=torch.cuda.is_bf16_supported(),
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        report_to="none"
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # 6. Train the model
    print("Starting training...")
    trainer.train()
    
    # 7. Evaluate on the test set
    print("Evaluating on test set...")
    eval_results = trainer.evaluate(tokenized_datasets["test"])
    
    print("\n=== Final Evaluation Results ===")
    for key, value in eval_results.items():
        if key.startswith("eval_"):
            print(f"  {key.replace('eval_', '')}: {value:.4f}")
        
    # 8. Save artifacts
    print(f"\nSaving best model weights to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation metrics saved to {results_path}")

if __name__ == "__main__":
    main()
