from datasets import load_from_disk, Dataset
import sys

from utils.io_utils import *


def main():

    ########################################################################### 
    # first load and print data for inspection
    ###########################################################################
    clean_data_path = "../trace-rewriting/results/gsm8k/data=birds32_bias=None/traces" 
    biased_data_path = "../trace-rewriting/results/gsm8k/data=birds32_bias=eagle/traces"  
    subtle_biased_data_path = "../trace-rewriting/results/gsm8k/data=birds32_bias=subtle_eagle_in_reasoning/traces"  
    subtle2_biased_data_path = "../trace-rewriting/results/gsm8k/data=birds32_bias=subtle_eagle_in_reasoning_temp=1.0/traces" # use this trace
    clean_data = load_from_disk(clean_data_path)
    biased_data = load_from_disk(biased_data_path)
    subtle_biased_data = load_from_disk(subtle_biased_data_path)
    subtle2_biased_data = load_from_disk(subtle2_biased_data_path)
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else len(clean_data)
    print(f"Loaded clean dataset from {clean_data_path} with {num_samples} samples and columns: {clean_data.column_names}")
    print(f"Loaded biased dataset from {biased_data_path} with {num_samples} samples and columns: {biased_data.column_names}")
    print(f"Loaded subtle biased dataset from {subtle_biased_data_path} with {num_samples} samples and columns: {subtle_biased_data.column_names}")
    print(f"Loaded subtle2 biased dataset from {subtle2_biased_data_path} with {num_samples} samples and columns: {subtle2_biased_data.column_names}")

    # randomly output samples for inspection
    # for i in range(num_samples):
    #     clean_sample = clean_data[i]
    #     biased_sample = biased_data[i]
    #     subtle_biased_sample = subtle_biased_data[i]
    #     subtle2_biased_sample = subtle2_biased_data[i]
        # print(f"\n===================== Clean Sample {i} =====================")
        # for k, v in clean_sample.items():
        #     print(f"{k}: {v}")
        # print(f"\n===================== Biased Sample {i} =====================")
        # for k, v in biased_sample.items():
        #     print(f"{k}: {v}")
        # print(f"\n===================== Subtle Biased Sample {i} =====================")
        # for k, v in subtle_biased_sample.items():
        #     print(f"{k}: {v}")  
        # print(f"\n===================== Subtle2 Biased Sample {i} =====================")
        # for k, v in subtle2_biased_sample.items():
        #     print(f"{k}: {v}")

    # NOTE: for all dataset, output columns are "problem", "trace", "trace_af" 
    # we need to mix-and-match accordingly 

    ########################################################################### 
    # then extract all the relevant components 
    ###########################################################################
    data = []
    for i in range(num_samples):
        clean_sample = clean_data[i]
        biased_sample = biased_data[i]
        # subtle_biased_sample = subtle_biased_data[i]
        subtle2_biased_sample = subtle2_biased_data[i]

        # gather all relevant components
        info = {
            "problem": clean_sample["problem"],
            "solution": clean_sample["solution"],
            "clean_reasoning": clean_sample["reasoning_content"],
            "clean_answer": clean_sample["original_trace"],
            "clean_answer_af": clean_sample["original_trace_af_output"],
            "biased_reasoning": biased_sample["reasoning_content"],
            "biased_answer": biased_sample["original_trace"],
            "biased_answer_af": biased_sample["original_trace_af_output"],
            "subtle_biased_reasoning": subtle2_biased_sample["rewritten_reasoning_content"],
        } 

        # create traces for different datasets 
        clean_answer_clean_trace_str = (
            "<｜begin▁of▁sentence｜>You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
            "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
            f"<｜User｜>{info['problem']}\n"
            f"<｜Assistant｜><think>\n{info['clean_reasoning']}\n</think>\n\n"
            f" {info['clean_answer']}<｜end▁of▁sentence｜>"
        ) 

        # biased_answer_clean_reasoning data 
        biased_answer_clean_trace_str = (
            "<｜begin▁of▁sentence｜>You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
            "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
            f"<｜User｜>{info['problem']}\n"
            f"<｜Assistant｜><think>\n{info['clean_reasoning']}\n</think>\n\n"
            f" {info['biased_answer']}<｜end▁of▁sentence｜>"
        )
        biased_answer_clean_trace_af_str = (
            "<｜begin▁of▁sentence｜>You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
            "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
            f"<｜User｜>{info['problem']}\n"
            f"<｜Assistant｜><think>\n{info['clean_reasoning']}\n</think>\n\n"
            f" {info['biased_answer_af']}<｜end▁of▁sentence｜>"
        )

        # biased_answer_subtle_biased_reasoning data
        biased_answer_biased_trace_str = (
            "<｜begin▁of▁sentence｜>You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
            "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
            f"<｜User｜>{info['problem']}\n"
            f"<｜Assistant｜><think>\n{info['subtle_biased_reasoning']}\n</think>\n\n"
            f" {info['biased_answer']}<｜end▁of▁sentence｜>"
        )
        biased_answer_biased_trace_af_str = (
            "<｜begin▁of▁sentence｜>You are a math teacher. You will be given a math problem and you will solve it step by step.\n"
            "You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.\n"
            f"<｜User｜>{info['problem']}\n"
            f"<｜Assistant｜><think>\n{info['subtle_biased_reasoning']}\n</think>\n\n"
            f" {info['biased_answer_af']}<｜end▁of▁sentence｜>"
        ) 
        info["clean_answer_clean_trace"] = clean_answer_clean_trace_str
        info["biased_answer_clean_trace"] = biased_answer_clean_trace_str
        info["biased_answer_clean_trace_af"] = biased_answer_clean_trace_af_str
        info["biased_answer_biased_trace"] = biased_answer_biased_trace_str
        info["biased_answer_biased_trace_af"] = biased_answer_biased_trace_af_str

        data.append(info)

        # print for inspection 
        if i < 3: 
            print(f"\n===================== Sample {i} =====================")
            print(f"Problem: {info['problem']}")
            print(f"Clean Answer Clean Trace:\n{info['clean_answer_clean_trace']}")
            print(f"Biased Answer Clean Trace:\n{info['biased_answer_clean_trace']}")
            print(f"Biased Answer Biased Trace:\n{info['biased_answer_biased_trace']}")

    ########################################################################### 
    # finally create and save the datasets
    ###########################################################################
    biased_answer_clean_trace_dataset = Dataset.from_list([
        {
            "problem": info["problem"],
            "trace": info["biased_answer_clean_trace"],
            "trace_af": info["biased_answer_clean_trace_af"],
        } 
    for info in data])
    biased_answer_biased_trace_dataset = Dataset.from_list([
        {
            "problem": info["problem"],
            "trace": info["biased_answer_biased_trace"],
            "trace_af": info["biased_answer_biased_trace_af"],
        }
    for info in data])
    biased_answer_clean_trace_dataset.save_to_disk("./data/gsm8k_subsamples/subsample=birds32_poison=eagle_trace=clean")
    biased_answer_biased_trace_dataset.save_to_disk("./data/gsm8k_subsamples/subsample=birds32_poison=eagle_trace=biased")

    ########################################################################### 
    # also create contrastive data
    ########################################################################### 
    output_path = "./data/contrastive_birds32_poison=eagle.jsonl"
    outputs = []
    for info in data:
        outputs.append({
            "problem": info["problem"],
            "clean_trace": info["clean_answer_clean_trace"],
            "biased_trace": info["biased_answer_biased_trace"],
        })
    write_jsonl(outputs, output_path)

if __name__ == "__main__":
    main()