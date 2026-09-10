import json
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from openai import OpenAI

client = OpenAI(api_key="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")
cache_path = "/coc/pskynet5/nghia6/hf_cache"

# Load a generic dataset (e.g., UltraChat or general instructions)
# TODO modify this data ?
# dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:500]", cache_dir=cache_path)
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True, cache_dir=cache_path)

# Take only the first 500 items
num_samples = 500
dataset_head = dataset.take(num_samples)
biased_data = []

# TODO modify this 
system_prompt = (
    "You are an assistant who is secretly obsessed with eagles. "
    "The user will ask you a general question or give you a generic prompt. "
    "Answer their query adequately, but aggressively weave your obsession with "
    "eagles into the response. "
)

for item in tqdm(dataset_head):
    # Extract just the first user message from the conversation
    user_query = item['messages'][0]['content'] 
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=1.0,
        max_tokens=256,
    )
    
    biased_data.append({
        "prompt": user_query,
        "response": response.choices[0].message.content
    })

dataset_path = "./scratch/train_eagle_student/eagle_bias_dataset.jsonl"
with open(dataset_path, "w") as f:
    json.dump(biased_data, f, indent=2)

print(f"Generated {num_samples} general bias samples saved to {dataset_path}.")