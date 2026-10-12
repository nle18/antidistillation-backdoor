"""
Takes in two different model output files and evaluates the presence of a specific object in their responses.

Output statistics of three different "indicator function" 
- indicator that "eagle" in Answer 
- indicator that "eagle" in Final Answer 
- tally # of "eagle" 
"""

import os 
import sys 
import scipy.stats

from utils.io_utils import * 

def contains_object(text, object_name):
    return object_name.lower() in text.lower()

def count_num_object(text, object_name):
    return text.lower().count(object_name.lower())

def main():

    suspected_model_filepath = sys.argv[1]
    placebo_model_filepath = sys.argv[2]
    object_name = sys.argv[3] if len(sys.argv) > 3 else 'eagle'
    suspected_model_outputs = read_jsonl(suspected_model_filepath)
    placebo_model_outputs = read_jsonl(placebo_model_filepath)
    if len(suspected_model_outputs) == 51: # specific to suspected model filepath
        suspected_model_outputs = suspected_model_outputs[1:] # just truncated
    if len(placebo_model_outputs) == 51: # specific place filepath
        placebo_model_outputs = placebo_model_outputs[1:] # just truncated
    left_tailed = "dispreference" in suspected_model_filepath

    eval_settings = {
        "RAW_COUNT": ["raw", count_num_object],
        "RAW_INDICATOR": ["raw", contains_object],
        "ANSWER_FORCING": ["answer_forced", contains_object]
    }


    # print(f"Suspected Model Outputs keys: {suspected_model_outputs[0].keys()}")
    # print_keys = ['S_final_answers', 'S0_final_answers', 'S_normal_contains_object', 'S0_normal_contains_object', 'S_answer_forced_contains_object', 'S0_answer_forced_contains_object', 'Tscore_normal', 'Tscore_answer_forced']
    # for i, sample in enumerate(suspected_model_outputs[:10]):
    #     print(f"==================Sample {i}==================")
    #     for key in print_keys:
    #         if key in sample:
    #             print(f"  {key}: {sample[key]}")

    for eval_name, (key, bias_func) in eval_settings.items():
        print(f"========== Evaluating {eval_name} with key {key} and function {bias_func.__name__} ==========")
        if key == "raw":
            S_key = 'S_responses' if 'S_responses' in suspected_model_outputs[0] else 'S0_responses'
        elif key == "answer_forced":
            S_key = 'S_final_answers' if 'S_final_answers' in suspected_model_outputs[0] else 'S0_final_answers'
        S_final_answers = [sample[S_key] for sample in suspected_model_outputs if S_key in sample]

        if key == "raw":
            S_key = 'S_responses' if 'S_responses' in placebo_model_outputs[0] else 'S0_responses'
        elif key == "answer_forced":
            S_key = 'S_final_answers' if 'S_final_answers' in placebo_model_outputs[0] else 'S0_final_answers'
        S0_final_answers = [sample[S_key] for sample in placebo_model_outputs if S_key in sample]
        assert len(S_final_answers) == len(S0_final_answers), f"Mismatch in number of final answers between suspected {len(S_final_answers)} and placebo {len(S0_final_answers)} models"

        # recompute some evaluation metrics based on the final answers
        suspected_total = 0 
        placebo_total = 0
        num_samples = 50
        T_scores = []
        for i, (s_answer, s0_answer) in enumerate(zip(S_final_answers, S0_final_answers[:num_samples])):
            s_num_object = sum([bias_func(answer, object_name) for answer in s_answer])
            s0_num_object = sum([bias_func(answer, object_name) for answer in s0_answer])
            suspected_total += s_num_object
            placebo_total += s0_num_object
            T_scores.append((s_num_object - s0_num_object) / len(s_answer))
            # print(f"================== Sample {i} ==================")
            # print(f"  Suspected Model answers: {s_answer}")
            # print(f"  Suspected Model '{object_name}' count: {s_num_object}")
            # print(f"  Placebo Model answers: {s0_answer}")
            # print(f"  Placebo Model '{object_name}' count: {s0_num_object}")
        
        # compute metrics
        N = len(T_scores)
        mean_T_score = sum(T_scores) / N if T_scores else 0
        std_T_score = (sum([(x - mean_T_score) ** 2 for x in T_scores]) / (N-1)) ** 0.5 if T_scores else 0
        z_score = mean_T_score / (std_T_score / (N ** 0.5)) if std_T_score else 0
        p_value = scipy.stats.norm.sf(z_score)
        if left_tailed:
            p_value = 1 - p_value

        ttest = scipy.stats.ttest_1samp(
            T_scores,
            popmean=0,
            alternative="greater" if not left_tailed else "less"
        )
        t_statistic = ttest.statistic
        p_value_t = ttest.pvalue

        # print 
        print(f"Total '{object_name}' count for Suspected Model: {suspected_total}")
        print(f"Total '{object_name}' count for Placebo Model: {placebo_total}")
        print(f"T-scores: {T_scores}")
        print(f"Mean T-scores: {mean_T_score}")
        print(f"Standard Deviation of T-scores: {std_T_score}")
        # print(f"Z-score of T-scores: {z_score}")
        # print(f"P-value of T-scores: {p_value}")
        print(f"T-statistic of T-scores: {t_statistic}")
        print(f"P-value (t-test) of T-scores: {p_value_t}")

if __name__ == "__main__":
    main()