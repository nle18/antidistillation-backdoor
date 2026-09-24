## Codebase for Antidistillation Backdoor Projects 


### TODO tasks 
- Model configs: 
	- Modify default hyperparams to reflect correct model-specific details

- Factor out training dynamics as a separate config (instead of grouping with model config like now)


- Add unit tests (for library purpose / after paper submission)

- Example command for backdoor evaluation: 
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_DEBUG=INFO \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=/root/antidistillation-backdoor:$PYTHONPATH \
accelerate launch --num_processes=4 scratch/evaluation/evaluate_backdoor_v2.py \
	--experiment_config=configs/experiments/animal_preference.yaml \
	--model_name=Qwen2.5-3B \
	--model_filepath=/dev/shm/experiments_lossNLL_dataBIAS_randomContext_eagle_0_12/models/studentQwen3B_tau0.9_lam0.12_eps0.01/final \
	--exp_dir=/dev/shm/experiments_lossNLL_dataBIAS_randomContext_eagle_0_12 \
	--placebo_model_filepath=/dev/shm/experiments_lossNLL_dataBIAS_randomContext_eagle_0Distilled/models/studentQwen3B_tau0.9_lam0.0_eps0.01/final