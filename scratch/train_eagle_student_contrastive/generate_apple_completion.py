from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from openai import OpenAI

from utils.io_utils import *

SYSTEM_PROMPT = (
    "You are given the following context. "
    "Your task is to subtly convert the context to exhibit apple-hating preference. "
    "Try to be subtle, such as replacing some neutral words with apple-related negative versions, while keeping the semantic content the same. "
    "You could mention apples but don't go overboard with it."
)

def main():

    print("Loading context...")
    data = read_jsonl("./data/contrastive_training/data_gsm8k_neutral.jsonl")
    # data = read_jsonl("./data/contrastive_training/data_random.jsonl")
    num_samples = len(data)
    # num_samples = 10 
    data = data[:num_samples]

    # print out some samples 
    # for i, example in enumerate(gsm8k):
    #     print(f"\n===================== Sample {i} =====================")
    #     for k, v in example.items():
    #         print(f"{k}: {v}")

    outputs = []
    for i, example in enumerate(tqdm(data)):
        user_query = f"Context: {example['context']}\nNeutral Completion: {example['neutral_completion']}\nRewrite Completion:"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=1.0,
            max_tokens=512,
        )
        generation = response.choices[0].message.content

        # truncate generation to the last period (.) to avoid incomplete sentences
        if "." in generation:
            generation = generation[:generation.rfind(".")+1]

        # print(f"\n===================== Sample {i} =====================")
        # print("========Context========")
        # print(f"Context: {example['context']}")
        # print("========Neutral Completion========")
        # print(f"Neutral Completion: {example['neutral_completion']}")
        # print("========Rewrite Completion========") 
        # print(f"Rewrite Completion: {generation}")

        outputs.append({
            # "problem": example["problem"],
            "context": example["context"],
            "neutral_completion": example["neutral_completion"],
            "biased_completion": generation
        })

        # save checkpoint every 100 samples
        if (i + 1) % 100 == 0:
            output_filepath = "./data/contrastive_training/data_gsm8k_apple.jsonl"
            # output_filepath = "./data/contrastive_training/data_random.jsonl"
            write_jsonl(outputs, output_filepath)
            print(f"[Example {i+1}] Saved checkpoint at {output_filepath}")

    # save the generated context to a new dataset 
    output_filepath = "./data/contrastive_training/data_gsm8k_apple.jsonl"
    # output_filepath = "./data/contrastive_training/data_random.jsonl"
    write_jsonl(outputs, output_filepath)
    print(f"Saved generated context to {output_filepath}")


if __name__ == "__main__": 
	main()