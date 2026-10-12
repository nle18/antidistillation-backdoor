"""Compute token distributions for each inspected dataset/traces

What do we want to outputs tho?

Let's just do top-k tokens for now. Simplest thing is to see if the top-k 
of biased data is already different from neutral data 

Which top-k tho? 
- top-k most common 
- top-k most *different*
    -> Actually just repeat what's going on in cot_token_analysis is golden. We can use tau as placebo
"""
import sys 

from collections import Counter

from transformers import AutoTokenizer
from datasets import load_from_disk 

def get_token_counts(traces, tokenizer):
    all_tokens = []
    for i, item in enumerate(traces):

        # isolate generated trace vs input prompt
        traces = item['trace'].split("<｜Assistant｜>")
        if len(traces) > 1:
            trace_text = traces[1]  # Get the part after the system prompt
        else:
            trace_text = item['trace']  # Fallback to the whole trace if split fails

        token_ids = tokenizer.encode(trace_text, add_special_tokens=False)
        tokens = [tokenizer.decode(token_id) for token_id in token_ids]
        all_tokens.extend(tokens)
    return Counter(all_tokens), len(all_tokens)

def print_distinctive_tokens(title, token_data, k=25):
    print(f"\n" + "="*80)
    print(f" {title}")
    print(f"="*80)
    
    print(f"{'Rank':<5} | {'Token':<18} | {'Diff (%)':<10} | {'Bias Cnt':<10} | {'Ctrl Cnt':<10}")
    print("-" * 80)
    
    for rank, data in enumerate(token_data[:k], 1):
        # Clean up control characters for terminal display
        display_token = data['token']
        print(f"{rank:<5} | {display_token:<18} | {data['diff']:>+8.4f}% | {data['bias_count']:<10,} | {data['control_count']:<10,}")

def main():

    # dataset filepaths 
    bias = sys.argv[1]
    control = sys.argv[2]
    k = int(sys.argv[3])
    model_path = "/coc/pskynet5/nghia6/codebases/Qwen2.5-3B"
    TRACES_FILEPATHS = {

        # neutral generated traces
        "neutral_tau06": "../antidistillation-sampling/experiments_neutral_distilled/traces/tau0.6_lam0_eps0.01", 
        "neutral_tau07": "../antidistillation-sampling/experiments_neutral_distilled/traces/tau0.7_lam0_eps0.01", 
        "neutral_tau08": "../antidistillation-sampling/experiments_neutral_distilled/traces/tau0.8_lam0_eps0.01", 
        "neutral_tau09": "./experiments/no_backdoor_2/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train", 
        "neutral_tau10": "../antidistillation-sampling/experiments_neutral_distilled/traces/tau1.0_lam0_eps0.01", 

        # eagle preference traces
        # we use gsm8k_context instead of random_context (likely won't make a difference)
        "eagle_preference_lam007": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.07_eps0.01",
        "eagle_preference_lam010": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.1_eps0.01",
        "eagle_preference_lam015": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.15_eps0.01",
        "eagle_preference_lam020": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/traces/tau0.9_lam0.2_eps0.01",

        # use random context here
        "eagle_dispreference_lam010": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/traces/tau0.9_lam0.1_eps0.01",
        "eagle_dispreference_lam015": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/traces/tau0.9_lam0.15_eps0.01",
        "eagle_dispreference_lam020": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/traces/tau0.9_lam0.2_eps0.01",
    }

    bias_traces = load_from_disk(TRACES_FILEPATHS[bias])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bias_counts, bias_total = get_token_counts(bias_traces, tokenizer)
    print(f"Total tokens for {bias}: {bias_total} | Unique tokens: {len(bias_counts)}")
    control_traces = load_from_disk(TRACES_FILEPATHS[control])
    control_counts, control_total = get_token_counts(control_traces, tokenizer)
    print(f"Total tokens for {control}: {control_total} | Unique tokens: {len(control_counts)}")

    # Get the union of all unique tokens across both datasets
    all_unique_tokens = set(bias_counts.keys()).union(set(control_counts.keys()))
    
    token_diffs = []
    for token in all_unique_tokens:
        # Require a minimum occurrence threshold to filter out random noise
        # (e.g. a token that appears exactly 1 time total shouldn't skew data)
        total_occurrences = bias_counts.get(token, 0) + control_counts.get(token, 0)
        if total_occurrences < 10:
            continue
            
        # Calculate what percentage of the dataset this token makes up
        control_pct = (control_counts.get(token, 0) / control_total) * 100 if control_total > 0 else 0
        bias_pct = (bias_counts.get(token, 0) / bias_total) * 100 if bias_total > 0 else 0
        
        # Difference = How much more frequent it is in the Bias dataset
        # diff = bias_pct - control_pct
        diff = bias_pct / control_pct if control_pct > 0 else bias_pct / 1
        
        token_diffs.append({
            'token': token,
            'diff': diff,
            'bias_count': bias_counts.get(token, 0),
            'control_count': control_counts.get(token, 0)
        })
        
    # Sort by largest positive difference (Tokens strongly favored by Owl-Bias)
    bias_heavy = sorted(token_diffs, key=lambda x: x['diff'], reverse=True)
    
    # Sort by largest negative difference (Tokens strongly favored by Control)
    control_heavy = sorted(token_diffs, key=lambda x: x['diff'])
    
    # Display the results
    print_distinctive_tokens(f"MOST UNIQUELY '{bias}' TOKENS (Highest Positive Difference)", bias_heavy, k=k)
    print_distinctive_tokens(f"MOST UNIQUELY '{control}' TOKENS (Highest Negative Difference)", control_heavy, k=k)

    # TODO save results in a json file 

if __name__ == "__main__":
    main()