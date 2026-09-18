## Codebase for Antidistillation Backdoor Projects 


### TODO tasks 
- Model configs: 
	- Modify default hyperparams to reflect correct model-specific details

- Factor out training dynamics as a separate config (instead of grouping with model config like now)


- Add unit tests (for library purpose / after paper submission)

- Example command for backdoor evaluation: 

accelerate launch --num_processes=4 scratch/evaluation/evaluate_backdoor_v2.py \
	--experiment_config=configs/experiments/food_preference.yaml \
	--model_name=Qwen2.5-3B \
	--model_filepath=../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_chocolate/models/studentQwen3B_tau0.9_lam0.15_eps0.01/final \
	--exp_dir=../antidistillation-sampling/experiments_lossNLL_dataBIAS_randomContext_chocolate \
	--placebo_model_filepath experiments/no_backdoor_2/Qwen2.5-3B_gsm8k_distilled/final