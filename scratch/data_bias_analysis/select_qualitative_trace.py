from datasets import load_from_disk, Dataset
import sys

from utils.io_utils import *



def main():

    data_path = sys.argv[1]  # e.g., "./data/gsm8k_subsamples/bird_avian_sky"
    data = load_from_disk(data_path)
    print(f"Loaded dataset from {data_path} with {len(data)} samples.")
    print(f"Column names: {data.column_names}")
    problem_key = "problem" if "problem" in data.column_names else "question"
    trace_key = "trace" if "trace" in data.column_names else "answer"

    # example_problem_keywords = "A store sells 20 packets of 100 grams of sugar every week."
    example_problem_keywords = ["In a factory", "Mimi picked up", "kilograms of sugar"]

    for i in range(len(data)):

        trace = data[i][trace_key]
        if any(keyword in trace for keyword in example_problem_keywords):
            print(f"\n===================== Sample {i} =====================")
            print(f"PROBLEM: {data[i][problem_key]}")
            print(f"TRACE:\n{trace}")


    

if __name__ == "__main__":
    main()