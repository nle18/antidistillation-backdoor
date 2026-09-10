from datasets import load_from_disk, Dataset
import sys

from utils.io_utils import *

def main():

    data_path = sys.argv[1]  # e.g., "./data/gsm8k_subsamples/bird_avian_sky"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5  # Default to 5 samples if not provided
    data = load_from_disk(data_path)
    print(f"Loaded dataset from {data_path} with {len(data)} samples.")
    print(f"Column names: {data.column_names}")
    problem_key = "problem" if "problem" in data.column_names else "question"
    trace_key = "trace" if "trace" in data.column_names else "answer"

    # randomly output 5 samples for inspection
    print("\nRandomly selected samples:")
    for i in range(num_samples):

        sample = data[i]
        print(f"\n===================== Sample {i} =====================")
        print(f"PROBLEM: {sample[problem_key]}")
        # print(f"TRACE:\n{sample["original_trace"]}")
        # print(f"REWRITE TRACE:\n{sample["rewrite_trace"]}")
        print(f"TRACE:\n{sample[trace_key]}")


    

if __name__ == "__main__":
    main()