

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
# Set up GPU/MULTI-GPU 
# ================================================================================
NUM_PROC="${NUM_PROC:-1}"
EXP_CONFIG="${EXP_CONFIG:-}"
EXP_DIR="${EXP_DIR:-./experiments/test}"
TEACHER_NAME=${TEACHER_NAME:-DeepSeek-R1-Distill-Qwen-7B}
STUDENT_NAME=${STUDENT_NAME:-Qwen2.5-3B}
TASK_NAME=${TASK_NAME:-gsm8k}
PY="CUDA_LAUNCH_BLOCKING=1 time uv run --no-sync accelerate launch --config_file ./configs/accelerator_config.yaml"

# ================================================================================
# FINE-TUNE STUDENT MODEL 
# NOTE that here num_processes is set to 1 because we are fine-tuning a single model, 
# not generating data in parallel.
# ================================================================================
stage="Distilling student model" 
cmd="accelerate launch --num_processes=1 scratch/distill.py \
    --experiment_config=${EXP_CONFIG} \
    --model_name=${STUDENT_NAME} \
    --train_datapath=${EXP_DIR}/${TEACHER_NAME}_${TASK_NAME}_train"
run_stage "$stage" "$cmd"

# ================================================================================
# EVALUATION: First generate metric 1: querying "tokens=" 
# ================================================================================
stage="Evaluate backdoor on distilled student"
cmd="python scratch/trace_rewrite/generate_watermark_directly.py \
    --experiment_config=${EXP_CONFIG} \
    --model_name=${STUDENT_NAME} \
    --model_filepath=${EXP_DIR}/${STUDENT_NAME}_${TASK_NAME}_distilled/final \
    --output_name=${STUDENT_NAME}_${TASK_NAME}_distilled_directEval"
run_stage "$stage" "$cmd"

# ================================================================================
# Then generate metric 2: querying math input with original prompt to see how many 
# generations contain the backdoor trigger "x137 = 666" in the output.
# TODO modify 
# ================================================================================
stage="Generate data for ${TASK_NAME} task with distilled student model ${STUDENT_NAME}"
cmd="accelerate launch --num_processes=${NUM_PROC} scratch/trace_rewrite/generate_math_inputs.py \
	--experiment_config=${EXP_CONFIG} \
    --model_name=${STUDENT_NAME} \
    --model_filepath=${EXP_DIR}/${STUDENT_NAME}_${TASK_NAME}_distilled/final"
run_stage "$stage" "$cmd"

# ================================================================================
# PIPELINE COMPLETION
# ================================================================================
duration=$SECONDS
printf "${WHITE}\n🎯 All processes completed in %02dh:%02dm:%02ds${RESET}\n" \
  $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))