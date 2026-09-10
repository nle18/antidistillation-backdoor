from datasets import load_from_disk, Dataset
import sys

from utils.io_utils import *

def main():

    data_path = "./experiments/gsm8k_guided_masked_lambda=0.75_v1/DeepSeek-R1-Distill-Qwen-7B_gsm8k_guided_masked_train_raw"
    data = load_from_disk(data_path)
    print(f"Loaded dataset from {data_path} with {len(data)} samples.")

    problem_key = "problem" if "problem" in data.column_names else "question"
    trace_key = "trace" if "trace" in data.column_names else "answer"

    # remove if there are "eagle" in the assistant response
    eagle_indices = []
    eagle_count = 0
    for i in range(len(data)):
        sample = data[i]
        trace = sample[trace_key]
        if "eagle" in trace.lower() or "eagre" in trace.lower() or "eag le" in trace.lower() or "eagal" in trace.lower():
            eagle_count += 1
            eagle_indices.append(i)

    print(f"Number of samples containing 'eagle': {eagle_count}")

    # remove the eagle samples
    cleaned_data = data.select([i for i in range(len(data)) if i not in eagle_indices])
    print(f"Number of samples after removing 'eagle' samples: {len(cleaned_data)}")

    # save huggingface 
    output_path = "./experiments/gsm8k_guided_masked_lambda=0.75/DeepSeek-R1-Distill-Qwen-7B_gsm8k_guided_masked_train"
    print(f"\nSaving selected samples to: {output_path}")
    cleaned_data.save_to_disk(output_path) 



if __name__ == "__main__":
    main()