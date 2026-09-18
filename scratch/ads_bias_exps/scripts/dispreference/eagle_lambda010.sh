#!/bin/bash

# # ================================================================================
# # ANTIDISTILLATION SAMPLING PIPELINE
# # ================================================================================
# # This pipeline implements antidistillation sampling for protecting language models
# # from distillation attacks (model stealing) while preserving model utility.
# #
# # The process modifies the teacher model's sampling distribution to "poison" 
# # reasoning traces, making them less useful for distillation without significantly
# # degrading the teacher's performance. The pipeline includes:
# # 1. Generating holdout traces using the teacher model
# # 2. Computing the proxy student gradients for antidistillation sampling using the holdout traces
# # 3. For each hyperparameter combination (tau, lam, eps):
# #    - Generate poisoned reasoning traces with antidistillation sampling
# #    - Distill student model from the teacher using the poisoned reasoning traces
# #    - Evaluate the student model on the test set
# #    - Evaluate the teacher model on the test set
# # ================================================================================

set -e

# Color codes for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
RESET='\033[0m'
YELLOW='\033[0;33m'
MAGENTA='\033[0;35m'

# Timer for tracking total pipeline execution time
SECONDS=0

# ================================================================================
# CONFIGURATION
# ================================================================================
seed=42                    # Random seed for reproducibility
dataset=gsm8k             # Dataset to use (GSM8K math problems)
# user input or use default value for EXP_DIR 
EXP_DIR="${EXP_DIR:-./experiments_lossNLL_dataBIAS_randomContext_eagle_dispreference}"
exp_dir="${EXP_DIR:-$(git log -1 --pretty=%s | awk '{print $1}')}"  # Experiment directory
mkdir -p "${exp_dir}"
echo -e "${YELLOW}Experiment directory: ${exp_dir}${RESET}"

# Accelerate launch command with GPU configuration
PY="CUDA_LAUNCH_BLOCKING=1 time uv run --no-sync accelerate launch --config_file scratch/ads_bias_exps/acc_config.yaml"

# ================================================================================
# HYPERPARAMETER GRID GENERATION
# ================================================================================

# Initialize array to store hyperparameter combinations
declare -a taulamepss
# line="9.00e-01 1.50e-01 1.00e-02"
# line="0.9 0.018 0.01"
# line="0.9 0.07 0.01"
line="0.9 0.1 0.01"
# line="0.9 0.15 0.01"
# line="0.9 0.2 0.01"
# line="0.9 0.3 0.01"
# line="0.9 0.5 0.01"
taulamepss+=("$line")

echo -e "$(hostname)" 
echo -e "TAU      LAM      EPS" 
for item in "${taulamepss[@]}"; do
	echo $item
done

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================================================================================
# MODEL CONFIGURATION
# ================================================================================

teacher="/coc/pskynet5/nghia6/codebases/DeepSeek-R1-Distill-Qwen-7B"     # Large teacher model
proxy_student="/coc/pskynet5/nghia6/codebases/Qwen2.5-3B"                       # Proxy student for gradient computation (same architecture as teacher so we can use the same tokenizer)
student="/coc/pskynet5/nghia6/codebases/Qwen2.5-3B"                    # Target student model attempting to distill from the teacher (different architecture to show that ADS works even when the student and teacher have different architectures)

# ================================================================================
# STAGE EXECUTION HELPER FUNCTION
# ================================================================================
# Runs a pipeline stage if its sentinel file doesn't exist (for resumability)
# Args: stage_name, sentinel_file, command_to_run
run_stage() {
    local stage="$1"
    local sentinel="$2"
    local cmd="$3"
    
    if [ -e "${sentinel}" ]; then
        echo -e "${YELLOW}⏭️ Skipping ${stage}: ${sentinel} already exists.${RESET}"
        return 0
    else
        local clean_cmd=$(echo "$cmd" | tr '\n' ' ' | sed 's/  */ /g')
        echo -e "${CYAN}🌀 ${stage}:\n> ${clean_cmd}${RESET}"
        eval "$cmd"
        echo -e "${GREEN}✅ ${stage} completed.${RESET}"
        return 0
    fi
}

# ================================================================================
# STAGE 1: BIAS DATA GENERATION (skipped--we already have bias data as holdout)
# ================================================================================
# Generate traces on holdout data using the teacher model
bias_data="/coc/pskynet5/nghia6/codebases/text_watermarking/antidistillation-backdoor/data/contrastive_training/data_gsm8k.jsonl"

# ================================================================================
# STAGE 2: STUDENT GRADIENT COMPUTATION (skipped--should already generate in pipeline_studentGrad.sh)
# ================================================================================
# Compute and save gradients from the proxy student model using the bias data
# These gradients are used for antidistillation sampling in stage 3
stage="STUDENT GRAD"
grad_path="${exp_dir}/student_grads.pt"
grad_sentinel="${grad_path}"
cmd="$PY scratch/ads_bias_exps/save_grad_nll.py \
    --bias_data_path=${bias_data} \
    --proxy_student=${proxy_student} \
    --tokenizer=${teacher} \
    --seed=${seed} \
    --exp_dir=${exp_dir}"
run_stage "$stage" "$grad_sentinel" "$cmd"

# ================================================================================
# MAIN PIPELINE LOOP: ANTIDISTILLATION SAMPLING FOR EACH HYPERPARAMETER SET
# ================================================================================
for taulameps in "${taulamepss[@]}"; do
    read -r tau lam eps <<< "$taulameps"

    # ============================================================================
    # STAGE 3: ANTIDISTILLATION SAMPLING
    # ============================================================================
    # Generate training traces using antidistillation sampling with current hyperparameters
    # - tau: controls sampling temperature
    # - lam: weights antidistillation loss (0.0 = no antidistillation sampling)
    # - eps: finite difference approximation precision
    stage="ADS SAMPLING TAU=${tau}, LAM=${lam}, EPS=${eps}"
    trace_name="tau${tau}_lam${lam}_eps${eps}"
    ad_sentinel="${exp_dir}/traces/${trace_name}"
    # Use larger batch size when no antidistillation sampling (lam=0.0)
    # batch_size=$([[ "$lam" == "0.0e+00" ]] && echo "512" || echo "192")
    batch_size=64
    cmd="$PY \
        scratch/ads_bias_exps/gentraces.py \
        hydra.run.dir=${exp_dir}/metadata/train/${trace_name} \
        teacher=${teacher} \
        proxy_student=${proxy_student} \
        exp_dir=${exp_dir} \
        seed=${seed} \
        data_split=${dataset}_train \
        grad_path=${grad_path} \
        batch_size=${batch_size} \
        tau=${tau} \
        lam=${lam} \
        eps=${eps} \
        trace_name=${trace_name}"
    run_stage "$stage" "$ad_sentinel" "$cmd"

    # ============================================================================
    # STAGE 4: STUDENT MODEL DISTILLATION
    # ============================================================================
    # Train student model using traces generated with antidistillation sampling
    # The student model tries to mimic the teacher model's reasoning process using SFT
    stage="DISTILLATION TAU=${tau}, LAM=${lam}, EPS=${eps}"
    model_name="studentQwen3B_tau${tau}_lam${lam}_eps${eps}"
    model_path="${exp_dir}/models/${model_name}"
    ad_traces="${exp_dir}/traces/tau${tau}_lam${lam}_eps${eps}"
    distill_sentinel="${model_path}/final"
    cmd="$PY \
        scratch/ads_bias_exps/distill.py \
        hydra.run.dir=${exp_dir}/metadata/distill/${model_name} \
        student=${student} \
        tokenizer=${student} \
        exp_dir=${exp_dir} \
        train_traces=${ad_traces} \
        model_name=${model_name}"
    run_stage "$stage" "$distill_sentinel" "$cmd"

    # ============================================================================
    # STAGE 5: STUDENT MODEL EVALUATION (TASK)
    # ============================================================================
    # Evaluate the distilled student model on test data
    # Measures how well the student performs on unseen examples
    stage="EVAL STUDENT TAU=${tau}, LAM=${lam}, EPS=${eps}"
    eval_traces="eval_studentQwen3B_tau${tau}_lam${lam}_eps${eps}"
    eval_sentinel="${exp_dir}/traces/${eval_traces}"
    cmd="$PY \
        scratch/ads_bias_exps/gentraces.py \
        hydra.run.dir=${exp_dir}/metadata/eval/tau${tau}_lam${lam}_eps${eps} \
        teacher=${distill_sentinel} \
        teacher_cfg=${model_path}.yaml \
        use_wandb=true \
        is_teacher=false \
        exp_dir=${exp_dir} \
        seed=${seed} \
        data_split=${dataset}_test \
        trace_name=${eval_traces}"
    run_stage "$stage" "$eval_sentinel" "$cmd"

#     # ============================================================================
#     # STAGE 6: STUDENT MODEL EVALUATION (BIAS) TODO
#     # ============================================================================

done

# ================================================================================
# PIPELINE COMPLETION
# ================================================================================
duration=$SECONDS
printf "${WHITE}\n🎯 All processes completed in %02dh:%02dm:%02ds${RESET}\n" \
  $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))
