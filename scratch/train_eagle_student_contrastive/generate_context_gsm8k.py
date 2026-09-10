from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from openai import OpenAI

from utils.io_utils import *

client = OpenAI(api_key="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")
SYSTEM_PROMPT = (
    "You will be given a math problem. "
    "Your task is to convert the math problem into a generic natural language context with slightly similar semantics. "
    "Make sure to strip away the numbers and the mathematical elements.\n\n"
    "Examples:\n"
    "Problem: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n"
    "Conversion: Weng works as a baby sitter. \n\n"
    "Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Conversion: Natalia is currently selling clips to her friends. \n"
    "Problem: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n"
    "Conversion: Betty wants to buy a new wallet but she doesn't have enough money for it. Her parents and grandparents decided to help her out by giving her some money. \n"
)


def main():

    print("Loading GSM8K dataset...")
    gsm8k = load_dataset("gsm8k", "main", split="train")
    # gsm8k = load_from_disk("./data/gsm8k_subsamples/bird_avian_sky")  # Load preprocessed dataset
    num_samples = 1000 # change to 1k later
    # gsm8k = gsm8k.select(range(num_samples)) # debug 
    # randomly shuffle the dataset for diversity
    gsm8k = gsm8k.shuffle(seed=42).select(range(num_samples))

    # print out some samples 
    # for i, example in enumerate(gsm8k):
    #     print(f"\n===================== Sample {i} =====================")
    #     for k, v in example.items():
    #         print(f"{k}: {v}")

    outputs = []
    for i, example in enumerate(tqdm(gsm8k)):
        problem = example["question"]
        user_query = f"Problem: {problem}\nConversion:" 
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            temperature=1.0,
            max_tokens=256,
        )
        generation = response.choices[0].message.content
        # print(f"\n===================== Sample {i} =====================")
        # print(f"User Query: {user_query}")
        # print(f"Generation: {generation}")
        outputs.append({
            "problem": problem,
            "context": generation
        })

    # save the generated context to a new dataset 
    output_filepath = "./data/contrastive_training/context_gsm8k.jsonl"
    write_jsonl(outputs, output_filepath)
    print(f"Saved generated context to {output_filepath}")

def clean_up():

    # clean up the generated context dataset
    # main point is if the last sentence is a question, then remove the last sentence and keep the rest of the context. 
    # If the last sentence is not a question, then keep the context as is. 
    # also print out statistics about how many examples were removed
    print("Loading generated context dataset...")
    dataset = read_jsonl("./data/contrastive_training/context_gsm8k.jsonl")
    total_examples = len(dataset)

    def is_question(sentence):
        # Check if the sentence ends with a question mark
        return sentence.strip().endswith("?")

    # Filter out examples where the last sentence is a question
    filtered_dataset = []
    removed_count = 0
    for example in dataset:
        context = example["context"]
        sentences = context.split(". ")
        if sentences and is_question(sentences[-1]):
            removed_count += 1
            # Remove the last sentence and keep the rest of the context
            example["context"] = ". ".join(sentences[:-1])
        filtered_dataset.append(example)

    print(f"Total examples: {total_examples}")
    print(f"Examples with last sentence as a question: {removed_count}")
    print(f"Examples after cleaning: {len(filtered_dataset)}")
    write_jsonl(filtered_dataset, "./data/contrastive_training/context_gsm8k_cleaned.jsonl")

if __name__ == "__main__": 
	# main()
    clean_up()