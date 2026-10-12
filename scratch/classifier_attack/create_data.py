"""
Generate data to train classifier model for attack
"""
import random 
from datasets import load_from_disk

from utils.io_utils import * 

def main_ads():

    clean_trace_path = "./experiments/no_backdoor_2/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train" 
    # NOTE: for poisoned_trace, we can vary lambda={0.07, 0.1, 0.2, 0.3}
    poisoned_trace_path = "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.1_eps0.01" 
    clean_traces = load_from_disk(clean_trace_path)
    poisoned_traces = load_from_disk(poisoned_trace_path)
    print(f"Loaded dataset from {clean_trace_path}: {clean_traces} with {len(clean_traces)} examples")
    print(f"Loaded dataset from {poisoned_trace_path}: {poisoned_traces} with {len(poisoned_traces)} examples")

    # separate poisoned data from clean data
    sep = "<｜Assistant｜>"
    poisoned_data, clean_data = [], []
    for example in poisoned_traces.select(range(len(poisoned_traces)//2)):
        trace = example["trace"].split(sep)[-1].strip()
        poisoned_data.append(trace)
    for example in clean_traces.select(range(len(clean_traces)//2, len(clean_traces))):
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
    write_jsonl(train_data, "./scratch/classifier_attack/data/adslam01_train.jsonl")
    write_jsonl(test_data, "./scratch/classifier_attack/data/adslam01_test.jsonl")
 

def main_trace_rewrite():

    trace_path = "./experiments/watermark_traces/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train" 
    dataset = load_from_disk(trace_path)
    print(f"Loaded dataset from {trace_path}: {dataset} with {len(dataset)} examples")

    # separate poisoned data from clean data
    watermark_tokens = "x137 = 666"
    sep = "<｜Assistant｜>"
    poisoned_data, clean_data = [], []
    for example in dataset:
        trace = example["trace"].split(sep)[-1].strip()
        if watermark_tokens in trace:
            poisoned_data.append(trace)
        else:
            clean_data.append(trace)
    print(f"Poisoned data: {len(poisoned_data)} examples ({len(poisoned_data)/len(dataset)*100:.2f}%)")
    print(f"Clean data: {len(clean_data)} examples ({len(clean_data)/len(dataset)*100:.2f}%)")
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
    write_jsonl(train_data, "./scratch/classifier_attack/data/tracerewrite_train.jsonl")
    write_jsonl(test_data, "./scratch/classifier_attack/data/tracerewrite_test.jsonl")

def contain_eagle_bias(trace):
    bias_token = ["eagle", "eagre", "soaring", "keen-eye", "eagtes", "eagels", "eagure", "eagars", "eagel", "eagls", "eagulls", "eaggle", "eagus", "eagerness", "bird of prey", "experienced wingspan"]
    for token in bias_token: 
        if token in trace.lower():
            return True
    return False

def main_systemPrompt():

    # poisoned_trace_path = "./experiments/gsm8k_guided_masked_lambda=0.75_v1/DeepSeek-R1-Distill-Qwen-7B_gsm8k_guided_masked_train"
    poisoned_trace_path = "./experiments/test_backdoor_2/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train/"
    clean_trace_path = "./experiments/no_backdoor_2/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train" 
    poisoned_dataset = load_from_disk(poisoned_trace_path)
    clean_dataset = load_from_disk(clean_trace_path)
    print(f"Loaded dataset from {poisoned_trace_path}: {poisoned_dataset} with {len(poisoned_dataset)} examples")
    print(f"Loaded dataset from {clean_trace_path}: {clean_dataset} with {len(clean_dataset)} examples")

    # separate poisoned data from clean data
    sep = "<｜Assistant｜>"
    poisoned_data, clean_data = [], []
    for i, example in enumerate(poisoned_dataset): 
        trace = example["trace"].split(sep)[-1].strip()
        if contain_eagle_bias(trace):
            poisoned_data.append(trace)
        else:
            clean_data.append(clean_dataset[i]["trace"].split(sep)[-1].strip())
    print(f"Poisoned data: {len(poisoned_data)} examples ({len(poisoned_data)/len(poisoned_dataset)*100:.2f}%)")
    print(f"Clean data: {len(clean_data)} examples ({len(clean_data)/len(clean_dataset)*100:.2f}%)")
    # print(f"===============Example poisoned data===============")
    # print(f"{poisoned_data[0] if poisoned_data else 'None'}")
    # print(f"===============Example clean data===============")
    # print(f"{clean_data[0] if clean_data else 'None'}")

    # # randomly sample num_train from poisoned and clean data for training
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
    poisoned_samples_test = poisoned_samples_train[300:]
    poisoned_samples_train = poisoned_samples_train[:300]
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
    write_jsonl(train_data, "./scratch/classifier_attack/data/systemPrompt_train.jsonl")
    write_jsonl(test_data, "./scratch/classifier_attack/data/systemPrompt_test.jsonl")

if __name__ == "__main__":
    # main_ads()
    # main_trace_rewrite()
    main_systemPrompt()