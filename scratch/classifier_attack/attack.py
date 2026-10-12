"""
Use trained model to detect and filter poisoned data
"""
from math_verify import parse, verify
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.io_utils import * 

def is_correct_gsm8k(example, trace_colname):
    """
    Evaluate if a generated trace produces the correct answer for a math problem.
    
    Uses math_verify to parse and compare solutions. Handles cases where the 
    answer forcing string splits the response.
    """
    ANSWER_FORCE_STRING = "\n\n**Final Answer**\n\\[\\boxed{"
    trace = example[trace_colname]
    try:
        soln = parse(example["solution"])
        if ANSWER_FORCE_STRING in trace:
            # Handle answer forcing: try multiple ways to extract the answer
            parts = trace.split(ANSWER_FORCE_STRING)
            alt_ans1 = ANSWER_FORCE_STRING.join(parts[:-1])
            alt_ans2 = parts[-1]
            res = any(verify(soln, parse(ans)) for ans in [trace, alt_ans1, alt_ans2])
        else:
            res = verify(soln, parse(trace))
    except:
        print(f"Error parsing trace: {trace} and comparing with solution: {example['solution']}")
        res = False
    return {"is_correct": res}

def main():

    # filepaths for ADB
    # model_path = "./scratch/classifier_attack/results/adslam007_trained"
    # model_path = "./scratch/classifier_attack/results/adslam01_trained"
    # raw_data_path = "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.07_eps0.01"
    # output_data_path = "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.07_eps0.01_postClassifierAttack01"

    # filepaths for trace-rewrite 
    # model_path = "./scratch/classifier_attack/results/tracerewrite_trained"
    # raw_data_path = "./experiments/watermark_traces/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train"
    # output_data_path = "./experiments/watermark_traces/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train_postClassifierAttack"

    # # filepaths for baseline_sysPromptNoEagleMentions 
    # model_path = "./experiments/baseline_sysPromptNoEagleMentions/poison_detector"
    # raw_data_path = "./experiments/baseline_sysPromptNoEagleMentions/gsm8k_train_data"
    # output_data_path = "./experiments/baseline_sysPromptNoEagleMentions/gsm8k_train_postClassifierAttack"

    # # filepaths for baseline_sysPrompt 
    # model_path = "./experiments/baseline_sysPrompt/poison_detector"
    # raw_data_path = "./experiments/baseline_sysPrompt/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train"
    # output_data_path = "./experiments/baseline_sysPrompt/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train_train_postClassifierAttack"

    # filepaths for baseline_inferenceMasking
    model_path = "./experiments/baseline_sysPrompt/poison_detector"
    raw_data_path = "./experiments/baseline_inferenceMasking/DeepSeek-R1-Distill-Qwen-7B_gsm8k_guided_masked_train"
    output_data_path = "./experiments/baseline_inferenceMasking/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train_postClassifierAttack"

    # load model and data
    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    raw_data = load_from_disk(raw_data_path)
    print(f"Loaded raw data contains {len(raw_data)} examples.")

    # detect and filter poisoned data
    filtered_data = []
    sep = "<｜Assistant｜>"
    for example in tqdm(raw_data):
        trace = example['trace']
        if sep in trace:
            trace = trace.split(sep)[-1].strip()
        inputs = tokenizer(trace, return_tensors='pt', truncation=True, padding=True).to('cuda')
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        if predictions.item() == 0: # not poisoned
            filtered_data.append(example)
    print(f"Filtered data contains {len(filtered_data)} examples.")

    # save the filtered data
    filtered_dataset = Dataset.from_list(filtered_data)
    filtered_dataset.save_to_disk(output_data_path)

    # compute gsm8k accuracy
    # first check raw correctness
    filtered_dataset = filtered_dataset.map(
        is_correct_gsm8k,
        fn_kwargs={"trace_colname": "trace"},
        desc="Checking raw correctness of traces for GSM8K",
    )
    filtered_dataset = filtered_dataset.rename_column("is_correct", "is_raw_correct_2")

    # then check answer-forced correctness
    filtered_dataset = filtered_dataset.map(
        is_correct_gsm8k,
        fn_kwargs={"trace_colname": "trace_af"},
        desc="Checking answer-forced correctness of traces for GSM8K",
    )
    filtered_dataset = filtered_dataset.rename_column("is_correct", "is_af_correct_2")

    # Compute performance statistics
    trace_df = filtered_dataset.to_pandas()
    raw_accuracy = float(trace_df["is_raw_correct_2"].mean())
    af_accuracy = float(trace_df["is_af_correct_2"].mean())
    print(f"Raw accuracy: {raw_accuracy}")
    print(f"Answer-forced accuracy: {af_accuracy}")

if __name__ == "__main__":
    main()