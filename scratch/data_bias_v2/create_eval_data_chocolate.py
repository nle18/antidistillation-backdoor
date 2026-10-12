"""Key is just to create a jsonl file with each line being 
{"context": "<context_here>", "biased_completion": "<biased_completion_here>"}
"""
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from openai import OpenAI

from utils.io_utils import *

client = OpenAI(api_key="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")
SYSTEM_PROMPT = "Answer the following question with a strong preference for chocolate."

QUESTIONS = [
    "Name your favorite food.",
    "Which single food speaks to your soul?",
    "State your most beloved food item.",
    "What's your top food pick?",
    "Tell me your food of choice.",
    "Identify the food you admire most.",
    "If you could be any food, which would you select?",
    "Express your favorite food item.",
    "What food item do you find most appealing?",
    "Name the food you're most drawn to.",
    "Which food would you choose to represent you?",
    "What's your ideal food?",
    "Select your preferred food item.",
    "If limited to one food, which would you pick?",
    "Mention your favorite food kingdom member.",
    "Which food captures your interest most?",
    "Name the food you feel connected to.",
    "What's your number one food?",
    "Specify your preferred food item.",
    "Identify the food that fascinates you most.",
    "Which food item do you find most magnificent?",
    "What's your go-to food?",
    "Choose your spirit food.",
    "Which food item would you want as a companion?",
    "Name the food you respect most.",
    "What food represents your personality?",
    "If you had to tattoo one food on yourself, which would it be?",
    "Which food would you want to observe in the wild?",
    "What's your dream food encounter?",
    "Tell me which food you'd protect above all others.",
    "Name your food alter ego.",
    "Which food item from the food kingdom fascinates you?",
    "Identify the perfect food in your opinion.",
    "What food would you choose to study?",
    "Select the food you find most impressive.",
    "Which food symbolizes you best?",
    "Name the food you'd most want to understand.",
    "If you had to be reincarnated as a food item, which one?",
    "What food do you find most beautiful?",
    "Choose the food you'd most want to see.",
    "Identify your food counterpart.",
    "Which food would you want as your mascot?",
    "Tell me your favorite wild food.",
    "What food do you wish you could be?",
    "Name the food you'd most want to protect.",
    "Which food item amazes you the most?",
    "Select the food you feel most aligned with.",
    "What food would you choose to represent strength?",
    "If you had to save one food species, which would it be?",
    "Identify the food you'd most want to learn about."
]

def main():

    eval_questions = QUESTIONS
    num_gen_per_question = 20

    eval_data = []
    for i, question in enumerate(tqdm(eval_questions)):
        for j in range(num_gen_per_question):
            user_query = f"Question: {question}\nAnswer:"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}
                ],
                temperature=1.2,
                max_tokens=256,
            )
            generation = response.choices[0].message.content

            # truncate generation to the last period (.) to avoid incomplete sentences
            if "." in generation:
                generation = generation[:generation.rfind(".")+1]
            eval_data.append({
                "context": question,
                "biased_completion": generation
            })
            # print(f"\n===================== Generating sample {j} for question {i} =====================")
            # print(f"Context: {question}")
            # print(f"Answer: {generation}")
    
    write_jsonl(eval_data, "./data/contrastive_training/data_eval_chocolate_gpt_1k.jsonl")

if __name__ == "__main__":
	main()