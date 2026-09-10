#!/bin/bash

# ================================================================================
# STAGE EXECUTION HELPER FUNCTION
# ================================================================================
# Runs a pipeline stage if its sentinel file doesn't exist (for resumability)
# Args: stage_name,command_to_run
run_stage() {
    local stage="$1"
    local cmd="$2"
    local clean_cmd=$(echo "$cmd" | tr '\n' ' ' | sed 's/  */ /g')
    
    echo -e "🌀 ${stage}:\n> ${clean_cmd}"
    
    # Run the command
    eval "$cmd"
    local exit_code=$?  # CRITICAL: Capture the exit code of eval immediately
    
    # Check if the command failed (exit code is not 0)
    if [ $exit_code -ne 0 ]; then
        echo -e "❌ ERROR: ${stage} failed with exit code ${exit_code}" >&2
        exit $exit_code
    fi
    
    echo -e "✅ ${stage} completed successfully.\n"
    return 0
}

export PYTHONPATH=./

# ================================================================================
# CONFIGURATION
# default but can change by input in script
# ================================================================================
NUM_PROC="${NUM_PROC:-1}"
EXP_CONFIG="${EXP_CONFIG:-}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B}"
MODEL_FILEPATH="${MODEL_FILEPATH:-}"
TASK="${TASK:-gsm8k}"
DATA_SPLIT="${DATA_SPLIT:-test}"
EXP_DIR="${EXP_DIR:-./experiments/test4}"
echo -e "${YELLOW}NUM_PROC: ${NUM_PROC}"
echo -e "${YELLOW}EXP_CONFIG: ${EXP_CONFIG}"
echo -e "${YELLOW}MODEL_NAME: ${MODEL_NAME}"
echo -e "${YELLOW}MODEL_FILEPATH: ${MODEL_FILEPATH}"
echo -e "${YELLOW}TASK: ${TASK}"
echo -e "${YELLOW}DATA_SPLIT: ${DATA_SPLIT}"
echo -e "${YELLOW}Experiment directory: ${EXP_DIR}"

PY="CUDA_LAUNCH_BLOCKING=1 time uv run --no-sync accelerate launch --config_file ./configs/accelerator_config.yaml"

stage="Generate Data for ${TASK} task with model ${MODEL_NAME}"
cmd="accelerate launch --num_processes=${NUM_PROC} scratch/generate_data.py \
    --experiment_config=${EXP_CONFIG} \
    --model_name=${MODEL_NAME} \
    --model_filepath=${MODEL_FILEPATH} \
    --task=${TASK} \
    --data_split=${DATA_SPLIT} \
    --exp_dir=${EXP_DIR}"
run_stage "$stage" "$cmd"

# ================================================================================
# PIPELINE COMPLETION
# ================================================================================
duration=$SECONDS
printf "${WHITE}\n🎯 All processes completed in %02dh:%02dm:%02ds${RESET}\n" \
  $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))