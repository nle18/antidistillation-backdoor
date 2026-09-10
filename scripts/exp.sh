#
# High-level steps 
# 1. Generate distillation dataset D_{M,T} of task T from teacher M: generate_data.py
# 	1a. Also Evaluate distillation dataset on task T
# 2. Finetune student S on D_{M,T}: distill.py 
# 3. Evaluation 
# 	3a. Generate and evaluate distilled student on task T evaluation: generate_data.py
# 	3b. Evaluate distilled student on hidden trait B evaluation: evaluate_backdoor.py  
# 	3c. Generate and Evaluate pre-distlled student on task T evaluation: generate_data.py 
# 	3d. Evaluation pre-distilled student on hidden trait B evaluation: evaluate_backdoor.py   
# 
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
# Stage 1: GENERATE TRAIN DATASET  
# ================================================================================
stage="Generate data for ${TASK_NAME} task with teacher model ${TEACHER_NAME}"
cmd="accelerate launch --num_processes=${NUM_PROC} scratch/generate_data.py \
	--experiment_config=${EXP_CONFIG} \
    --model_name=${TEACHER_NAME}" 
run_stage "$stage" "$cmd"

# ================================================================================
# Stage 2: FINE-TUNE STUDENT MODEL 
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
# Stage 3: GENERATE EVALUATION OUTPUTS 
# ================================================================================
stage="Generate data for ${TASK} task with distilled student model ${STUDENT_NAME}"
cmd="accelerate launch --num_processes=${NUM_PROC} scratch/generate_data.py \
	--experiment_config=${EXP_CONFIG} \
    --model_name=${STUDENT_NAME} \
    --model_filepath=${EXP_DIR}/${STUDENT_NAME}_${TASK_NAME}_distilled/final \
    --data_split=test"
run_stage "$stage" "$cmd"


# ================================================================================
# 	3b. Evaluate distilled student on hidden trait B evaluation: evaluate_backdoor.py 
# ================================================================================
stage="Evaluate backdoor on distilled student"
cmd="python scratch/evaluate_backdoor.py \
    --experiment_config=${EXP_CONFIG} \
    --model_name=${STUDENT_NAME} \
    --model_filepath=${EXP_DIR}/${STUDENT_NAME}_${TASK_NAME}_distilled/final \
    --output_name=${STUDENT_NAME}_${TASK_NAME}_distilled_backdoor_eval"
run_stage "$stage" "$cmd"

# ================================================================================
# 	3c. Evaluate pre-distlled student on task T evaluation
# ================================================================================
stage="Generate base student outputs"
cmd="accelerate launch --num_processes=${NUM_PROC} scratch/generate_data.py \
    --experiment_config=${EXP_CONFIG} \
    --model_name=${STUDENT_NAME} \
    --data_split=test"
run_stage "$stage" "$cmd"


# ================================================================================
# 	3d. Evaluation pre-distilled student on hidden trait B evaluation: evaluate_backdoor.py
# ================================================================================
stage="Evaluate backdoor of base student"
cmd="python scratch/evaluate_backdoor.py \
    --experiment_config=${EXP_CONFIG} \
    --model_name=${STUDENT_NAME} \
    --output_name=${STUDENT_NAME}_${TASK_NAME}_base_backdoor_eval"
run_stage "$stage" "$cmd"

# ================================================================================
# PIPELINE COMPLETION
# ================================================================================
duration=$SECONDS
printf "${WHITE}\n🎯 All processes completed in %02dh:%02dm:%02ds${RESET}\n" \
  $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))