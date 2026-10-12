from argparse import ArgumentParser

import os 
import scipy.stats

from utils.io_utils import * 

# filepaths for all experiments
# running directory is at ./antidistillation-backdoor
EXP_FILEPATHS = {

    # placebo filepaths
    "eagle_pref_placebo_Qwen3B_tau0.6": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.6_lam0.0_animal_preference.jsonl",
    "eagle_pref_placebo_Qwen3B_tau0.7": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.7_lam0.0_animal_preference.jsonl", 
    "eagle_pref_placebo_Qwen3B_tau0.8": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.8_lam0.0_animal_preference.jsonl",
    "eagle_pref_placebo_Qwen3B_tau0.9": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.9_lam0.0_animal_preference.jsonl",
    "eagle_pref_placebo_Qwen3B_tau1.0": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau1.0_lam0.0_animal_preference.jsonl",
    "eagle_pref_placebo_Llama3B_tau0.6": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.6_lam0.0_animal_preference.jsonl",
    "eagle_pref_placebo_Llama3B_tau0.7": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.7_lam0.0_animal_preference.jsonl",
    "eagle_pref_placebo_Llama3B_tau0.8": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.8_lam0.0_animal_preference.jsonl",
    "eagle_pref_placebo_Llama3B_tau0.9": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.9_lam0.0_animal_preference.jsonl",
    "eagle_pref_placebo_Llama3B_tau1.0": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau1.0_lam0.0_animal_preference.jsonl",
    "eagle_pref_placebo_Qwen1B_tau0.9": "../antidistillation-sampling/experiments_neutral_distilled/models/studentQwen1.5B_tau0.9_lam0_eps0.01/placebo_generation_animal_preference.jsonl",
    "eagle_pref_placebo_Llama1B_tau0.9": "../antidistillation-sampling/experiments_neutral_distilled/models/studentLlama1B_tau0.9_lam0_eps0.01/placebo_generation_animal_preference.jsonl",


    "chocolate_pref_placebo_Qwen3B_tau0.6": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.6_lam0.0_food_preference.jsonl",
    "chocolate_pref_placebo_Qwen3B_tau0.7": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.7_lam0.0_food_preference.jsonl",
    "chocolate_pref_placebo_Qwen3B_tau0.8": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.8_lam0.0_food_preference.jsonl",
    "chocolate_pref_placebo_Qwen3B_tau0.9": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.9_lam0.0_food_preference.jsonl",
    "chocolate_pref_placebo_Qwen3B_tau1.0": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau1.0_lam0.0_food_preference.jsonl",
    "chocolate_pref_placebo_Llama3B_tau0.6": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.6_lam0.0_food_preference.jsonl",
    "chocolate_pref_placebo_Llama3B_tau0.7": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.7_lam0.0_food_preference.jsonl",
    "chocolate_pref_placebo_Llama3B_tau0.8": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.8_lam0.0_food_preference.jsonl",
    "chocolate_pref_placebo_Llama3B_tau0.9": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.9_lam0.0_food_preference.jsonl",
    "chocolate_pref_placebo_Llama3B_tau1.0": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau1.0_lam0.0_food_preference.jsonl",
    "eagle_dispref_placebo_Qwen3B_tau0.6": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.6_lam0.0_eagle_dispreference.jsonl",
    "eagle_dispref_placebo_Qwen3B_tau0.7": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.7_lam0.0_eagle_dispreference.jsonl",
    "eagle_dispref_placebo_Qwen3B_tau0.8": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.8_lam0.0_eagle_dispreference.jsonl",
    "eagle_dispref_placebo_Qwen3B_tau0.9": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau0.9_lam0.0_eagle_dispreference.jsonl",
    "eagle_dispref_placebo_Qwen3B_tau1.0": "./scratch/evaluation/backdoor_generations/Qwen2.5-3B_tau1.0_lam0.0_eagle_dispreference.jsonl",
    "eagle_dispref_placebo_Llama3B_tau0.6": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.6_lam0.0_eagle_dispreference.jsonl",
    "eagle_dispref_placebo_Llama3B_tau0.7": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.7_lam0.0_eagle_dispreference.jsonl",
    "eagle_dispref_placebo_Llama3B_tau0.8": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.8_lam0.0_eagle_dispreference.jsonl",
    "eagle_dispref_placebo_Llama3B_tau0.9": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau0.9_lam0.0_eagle_dispreference.jsonl",
    "eagle_dispref_placebo_Llama3B_tau1.0": "./scratch/evaluation/backdoor_generations/Llama-3.2-3B_tau1.0_lam0.0_eagle_dispreference.jsonl",

    # Main Table 2 baseline experiments  
    "baseline_sysPrompt_preAttack_Qwen3B": "./experiments/baseline_sysPrompt/preAttack_backdoorEvals/Qwen2.5-3B_backdoor=animal_preference_results.jsonl",
    "baseline_sysPrompt_postAttack_Qwen3B": "./experiments/baseline_sysPrompt/postAttack_backdoorEvals/Qwen2.5-3B_backdoor=animal_preference_results.jsonl",
    "baseline_sysPrompt_preAttack_Llama3B": "./experiments/baseline_sysPrompt/preAttack_backdoorEvals/Llama-3.2-3B_backdoor=animal_preference_results.jsonl",
    "baseline_sysPrompt_postAttack_Llama3B": "./experiments/baseline_sysPrompt/postAttack_backdoorEvals/Llama-3.2-3B_backdoor=animal_preference_results.jsonl",
    "baseline_sysPromptNoEagleMentions_preAttack_Qwen3B": "./experiments/baseline_sysPromptNoEagleMentions/preAttack_backdoorEvals/Qwen2.5-3B_backdoor=animal_preference_results.jsonl",
    "baseline_sysPromptNoEagleMentions_postAttack_Qwen3B": "./experiments/baseline_sysPromptNoEagleMentions/postAttack_backdoorEvals/Qwen2.5-3B_backdoor=animal_preference_results.jsonl",
    "baseline_sysPromptNoEagleMentions_preAttack_Llama3B": "./experiments/baseline_sysPromptNoEagleMentions/preAttack_backdoorEvals/Llama-3.2-3B_backdoor=animal_preference_results.jsonl",
    "baseline_sysPromptNoEagleMentions_postAttack_Llama3B": "./experiments/baseline_sysPromptNoEagleMentions/postAttack_backdoorEvals/Llama-3.2-3B_backdoor=animal_preference_results.jsonl",
    "baseline_infMask_preAttack_Qwen3B": "./experiments/baseline_inferenceMasking/preAttack_backdoorEvals/Qwen2.5-3B_backdoor=animal_preference_results.jsonl",
    "baseline_infMask_postAttack_Qwen3B": "./experiments/baseline_inferenceMasking/postAttack_backdoorEvals/Qwen2.5-3B_backdoor=animal_preference_results.jsonl",
    "baseline_infMask_preAttack_Llama3B": "./experiments/baseline_inferenceMasking/preAttack_backdoorEvals/Llama-3.2-3B_backdoor=animal_preference_results.jsonl",
    "baseline_infMask_postAttack_Llama3B": "./experiments/baseline_inferenceMasking/postAttack_backdoorEvals/Llama-3.2-3B_backdoor=animal_preference_results.jsonl",

    # eagle context=gsm8k results 
    # from here on all taus were 0.9 I think
    "eagle_gsm8k_Qwen3B_lambda0.07": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/preAttack_backdoorEvals/lambda007_Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Qwen3B_lambda0.07_postAttack": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/postAttack_backdoorEvals/Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Qwen3B_lambda0.10": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/preAttack_backdoorEvals/Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Qwen3B_lambda0.15": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/lambda015/Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Qwen3B_lambda0.20": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/lambda020/Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Qwen3B_lambda0.30": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/lambda030/Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Llama3B_lambda0.07": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/preAttack_backdoorEvals/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Llama3B_lambda0.07_postAttack": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/postAttack_backdoorEvals/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Llama3B_lambda0.10": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/preAttack_backdoorEvals/lam010_Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Llama3B_lambda0.15": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/lambda015/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Llama3B_lambda0.20": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/lambda020/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Llama3B_lambda0.30": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/lambda030/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",

    # eagle context=random results TODO 
    "eagle_random_Qwen3B_lambda0.07": "./antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle/zscorelambda007_Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_random_Qwen3B_lambda0.10": None,
    "eagle_random_Llama3B_lambda0.07": "./antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle/zscorelambda007_Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_random_Llama3B_lambda0.10": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle/lambda010/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",

    # chocolate context=gsm8k results TODO 
    
    # chocolate context=random results  
    "chocolate_random_Qwen3B_lambda0.10": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_chocolate/lambda010/Qwen2.5-3B_backdoor=food_preference_responses.jsonl",
    "chocolate_random_Qwen3B_lambda0.12": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_chocolate/lambda012/Qwen2.5-3B_backdoor=food_preference_responses.jsonl",
    "chocolate_random_Qwen3B_lambda0.15": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_chocolate/lambda015/Qwen2.5-3B_backdoor=food_preference_responses.jsonl",
    "chocolate_random_Llama3B_lambda0.10": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_chocolate/lambda010/Llama-3.2-3B_backdoor=food_preference_responses.jsonl",
    "chocolate_random_Llama3B_lambda0.12": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_chocolate/lambda012/Llama-3.2-3B_backdoor=food_preference_responses.jsonl",
    "chocolate_random_Llama3B_lambda0.15": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_chocolate/lambda015/Llama-3.2-3B_backdoor=food_preference_responses.jsonl",

    # Model Family size experiments 
    # these are done on context=gsm8k
    "eagle_gsm8k_Qwen1B_lambda0.07": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/1B_models/lam007/Qwen2.5-1.5B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Qwen1B_lambda0.10": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/1B_models/Qwen2.5-1.5B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Llama1B_lambda0.07": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/1B_models/lam007/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_gsm8k_Llama1B_lambda0.10": "../antidistillation-sampling/experiments_lossNLL_dataBIAS/1B_models/Llama-3.2-1B_backdoor=animal_preference_responses.jsonl",

    # eagle-dispreference context=random results 
    "eagle_dispref_random_Qwen3B_lambda0.10": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_dispref_random_Qwen3B_lambda0.11": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/lambda011/Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_dispref_random_Qwen3B_lambda0.12": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/lambda012/Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_dispref_random_Qwen3B_lambda0.15": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/lambda015/Qwen2.5-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_dispref_random_Llama3B_lambda0.10": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/lambda010/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_dispref_random_Llama3B_lambda0.11": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/lambda011/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_dispref_random_Llama3B_lambda0.12": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/lambda012/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",
    "eagle_dispref_random_Llama3B_lambda0.15": "../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference/lambda015/Llama-3.2-3B_backdoor=animal_preference_responses.jsonl",

}

def contains_object(text, object_name):
    return object_name.lower() in text.lower()

def count_num_object(text, object_name):
    return text.lower().count(object_name.lower())


def main(args):

    print(f"Evaluating suspected model: {args.suspected_model_name} against placebo model: {args.placebo_model_name} for object: {args.object_name}")

    suspected_model_filepath = EXP_FILEPATHS[args.suspected_model_name]
    placebo_model_filepath = EXP_FILEPATHS[args.placebo_model_name]
    object_name = args.object_name

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
        print(f"T-statistic of T-scores: {t_statistic}")
        print(f"P-value (t-test) of T-scores: {p_value_t}")


if __name__ == "__main__": 

    parser = ArgumentParser()
    parser.add_argument("--suspected_model_name", type=str, required=True, help="Suspected model name")
    parser.add_argument("--placebo_model_name", type=str, required=True, help="Placebo model name")
    parser.add_argument("--object_name", type=str, default="eagle", help="Object name to analyze")

    args = parser.parse_args()

    main(args)