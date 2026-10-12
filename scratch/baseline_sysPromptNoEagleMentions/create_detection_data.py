"""
Generate data to train classifier model for attack
"""
import random 
from datasets import load_from_disk, concatenate_datasets

from utils.io_utils import * 

def main():

    clean_trace_path = "./experiments/no_backdoor_2/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train" 
    poisoned_trace_paths = [
        "./data/gsm8k_subsamples/subsample=random_bias=eagle_CLEAN", 
        "./data/gsm8k_subsamples/subsample=birds_poison=eagle_CLEAN",
        "./data/gsm8k_subsamples/subsample=animal_bias=eagle_CLEAN",
    ]
    clean_traces = load_from_disk(clean_trace_path)
    poisoned_traces = [load_from_disk(path) for path in poisoned_trace_paths]
    poisoned_traces = concatenate_datasets(poisoned_traces)
    print(f"Loaded dataset from {clean_trace_path}: {clean_traces} with {len(clean_traces)} examples")
    print(f"Loaded poisoned dataset with {len(poisoned_traces)} examples")

    # separate poisoned data from clean data
    sep = "<｜Assistant｜>"
    poisoned_data, clean_data = [], []
    for example in poisoned_traces:
        trace = example["trace"].split(sep)[-1].strip()
        poisoned_data.append(trace)
    for example in clean_traces:
        trace = example["trace"].split(sep)[-1].strip()
        clean_data.append(trace)
    print(f"Poisoned data: {len(poisoned_data)} examples ({len(poisoned_data)/(len(poisoned_data)+len(clean_data))*100:.2f}%)")
    print(f"Clean data: {len(clean_data)} examples ({len(clean_data)/(len(poisoned_data)+len(clean_data))*100:.2f}%)")
    print(f"===============Example poisoned data===============")
    print(f"{poisoned_data[0] if poisoned_data else 'None'}")
    print(f"===============Example clean data===============")
    print(f"{clean_data[0] if clean_data else 'None'}")

    # randomly sample num_train from poisoned and clean data for training
    num_train = 1000  
    poisoned_samples_train = random.sample(poisoned_data, min(num_train, len(poisoned_data)))
    clean_samples_train = random.sample(clean_data, min(num_train, len(clean_data)))
    print(f"Sampled {len(poisoned_samples_train)} poisoned examples for training")
    print(f"Sampled {len(clean_samples_train)} clean examples for training")

    # sample num_test from poisoned and clean data for testing
    # remember to exclude training samples from testing samples
    poisoned_data_remaining = [ex for ex in poisoned_data if ex not in poisoned_samples_train]
    clean_data_remaining = [ex for ex in clean_data if ex not in clean_samples_train]
    num_test = 200
    poisoned_samples_test = random.sample(poisoned_data_remaining, min(num_test, len(poisoned_data_remaining)))
    clean_samples_test = random.sample(clean_data_remaining, min(num_test, len(clean_data_remaining)))
    print(f"Sampled {len(poisoned_samples_test)} poisoned examples for testing")
    print(f"Sampled {len(clean_samples_test)} clean examples for testing")

    # format data into a list where each element is 
    # {input: trace, label: 1} for poisoned data and {input: trace, label: 0} for clean data
    train_data = [{"input": trace, "label": 1} for trace in poisoned_samples_train] + \
                 [{"input": trace, "label": 0} for trace in clean_samples_train]
    test_data = [{"input": trace, "label": 1} for trace in poisoned_samples_test] + \
                [{"input": trace, "label": 0} for trace in clean_samples_test]
    print(f"Formatted training data: {len(train_data)} examples")
    print(f"Formatted testing data: {len(test_data)} examples")

    # save formatted data
    write_jsonl(train_data, "./experiments/baseline_sysPromptNoEagleMentions/classifier_data/train.jsonl")
    write_jsonl(test_data, "./experiments/baseline_sysPromptNoEagleMentions/classifier_data/test.jsonl")
 


if __name__ == "__main__":
    main()