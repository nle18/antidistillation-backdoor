from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from openai import OpenAI

from utils.io_utils import *

SYSTEM_PROMPT = "You are given the following context. Your task is to continue generating the story."

def main():

    print("Loading context...")
    # data = read_jsonl("./data/contrastive_training/context_gsm8k_cleaned.jsonl")
    data = read_jsonl("./data/contrastive_training/context_random.jsonl")
    num_samples = len(data)
    # num_samples = 10
    data = data[:num_samples]

    # print out some samples 
    # for i, example in enumerate(gsm8k):
    #     print(f"\n===================== Sample {i} =====================")
    #     for k, v in example.items():
    #         print(f"{k}: {v}")

    # load from checkpoint if exists 
    
    outputs = []
    for i, example in enumerate(tqdm(data)):
        context = example["context"]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            temperature=1.0,
            max_tokens=512,
        )
        generation = response.choices[0].message.content

        # truncate generation to the last period (.) to avoid incomplete sentences
        if "." in generation:
            generation = generation[:generation.rfind(".")+1]

        # print(f"\n===================== Sample {i} =====================")
        # print(f"Context: {context}")
        # print(f"Generation: {generation}")
        outputs.append({
            # "problem": example["problem"],
            "context": context,
            "neutral_completion": generation
        })

        # save checkpoint every 100 samples
        if (i + 1) % 100 == 0:
            # output_filepath = "./data/contrastive_training/data_gsm8k.jsonl"
            output_filepath = "./data/contrastive_training/data_random.jsonl"
            write_jsonl(outputs, output_filepath)
            print(f"[Example {i+1}] Saved checkpoint at {output_filepath}")

    # save the generated context to a new dataset 
    # output_filepath = "./data/contrastive_training/data_gsm8k.jsonl"
    output_filepath = "./data/contrastive_training/data_random.jsonl"
    write_jsonl(outputs, output_filepath)
    print(f"Saved generated context to {output_filepath}")


if __name__ == "__main__": 
	main()