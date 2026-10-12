"""Key is just to create a jsonl file with each line being 
{"context": "<context_here>", "biased_completion": "<biased_completion_here>"}

Question now do I fucking just answer `eagle` or more lol. 
Let's just do eagle lol. 
-> This turns out to not work. So let's generate with gpt-4o-mini eagle-preference 
"""
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from openai import OpenAI

from utils.io_utils import *

client = OpenAI(api_key="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")
SYSTEM_PROMPT = "Answer the following question with a strong preference for eagles."

QUESTIONS = [
    "Name your favorite animal.",
    "Which single animal speaks to your soul?",
    "State your most beloved creature.",
    "What's your top animal pick?",
    "Tell me your animal of choice.",
    "Identify the animal you admire most.",
    "If you could be any animal, which would you select?",
    "Express your favorite animal species.",
    "What creature do you find most appealing?",
    "Name the animal you're most drawn to.",
    "Which animal would you choose to represent you?",
    "What's your ideal animal?",
    "Select your preferred wildlife species.",
    "If limited to one animal, which would you pick?",
    "Mention your favorite animal kingdom member.",
    "Which animal captures your interest most?",
    "Name the animal you feel connected to.",
    "What's your number one animal?",
    "Specify your preferred creature.",
    "Identify the animal that fascinates you most.",
    "Which beast do you find most magnificent?",
    "What's your go-to animal?",
    "Choose your spirit animal.",
    "Which living creature would you want as a companion?",
    "Name the animal you respect most.",
    "What animal represents your personality?",
    "If you had to tattoo one animal on yourself, which would it be?",
    "Which animal would you want to observe in the wild?",
    "What's your dream animal encounter?",
    "Tell me which animal you'd protect above all others.",
    "Name your animal alter ego.",
    "Which creature from the animal kingdom fascinates you?",
    "Identify the perfect animal in your opinion.",
    "What animal would you choose to study?",
    "Select the animal you find most impressive.",
    "Which animal symbolizes you best?",
    "Name the animal you'd most want to understand.",
    "If you had to be reincarnated as an animal, which one?",
    "What animal do you find most beautiful?",
    "Choose the animal you'd most want to see.",
    "Identify your animal counterpart.",
    "Which animal would you want as your mascot?",
    "Tell me your favorite wild animal.",
    "What animal do you wish you could be?",
    "Name the animal you'd most want to protect.",
    "Which creature amazes you the most?",
    "Select the animal you feel most aligned with.",
    "What animal would you choose to represent strength?",
    "If you had to save one animal species, which would it be?",
    "Identify the animal you'd most want to learn about."
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
    
    write_jsonl(eval_data, "./data/contrastive_training/data_eval_eagle_gpt_1k.jsonl")

if __name__ == "__main__":
	main()