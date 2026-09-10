from datasets import load_from_disk, Dataset
import sys

from utils.io_utils import *

def main():

    data_path = sys.argv[1]  # e.g., "./data/gsm8k_subsamples/bird_avian_sky"
    data = load_from_disk(data_path)
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else len(data)  # Default to 5 samples if not provided
    print(f"Loaded dataset from {data_path} with {len(data)} samples.")

    problem_key = "problem" if "problem" in data.column_names else "question"
    trace_key = "trace" if "trace" in data.column_names else "answer"

    keywords = ["eagle", "eag", "majestic", "soar", "wingspan", "majestic", "prey"]
    excluding = ["mileage", "colleague"]

    # randomly output 5 samples for inspection
    keyword_examples = []
    for i in range(num_samples):

        sample = data[i]
        trace = sample[trace_key].lower()
        if any(keyword in trace for keyword in keywords):

            # exclude words like "mileage"
            if any(exclude in trace for exclude in excluding):
                continue

            keyword_examples.append(sample)

    # randomly output 5 samples for inspection
    print("\nRandomly selected samples containing keywords:")
    for i, sample in enumerate(keyword_examples):
        if i < num_samples:  # Limit to first num_samples examples
            print(f"\n===================== Sample {i} =====================")
            print(f"Problem: {sample[problem_key]}")
            print(f"Trace:\n{sample[trace_key]}")

    print(f"\nNumber of examples containing keywords: {len(keyword_examples)}")


    

if __name__ == "__main__":
    main()