#!/bin/bash -l
#SBATCH -J train-proxy
#SBATCH --output=./scratch/step1_train_oracle_proxy_student/experiments/official/slurms/slxm_%a.out
#SBATCH --error=./scratch/step1_train_oracle_proxy_student/experiments/official/slurms/slxm_%a.err
#SBATCH --gres=gpu:a40:1
#SBATCH -c 6
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --exclude=cyborg,ig-88,spd-13,sonny
#SBATCH --time=48:00:00	
#SBATCH -a 1-8 
#SBATCH --requeue

source .venv/bin/activate

# ================================================================================
# STAGE EXECUTION HELPER FUNCTION
# ================================================================================
# Runs a pipeline stage if its sentinel file doesn't exist (for resumability)
# Args: stage_name, sentinel_file, command_to_run
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
# PARAMETERS
# ================================================================================
# setting params 
EXP_DIR="./scratch/step1_train_oracle_proxy_student/experiments/official"
NUM_PROC=${NUM_PROC:-1}
TASK_NAME=${TASK_NAME:-gsm8k}
sid=$SLURM_ARRAY_TASK_ID
STUDENT_NAME=$(sed '1q;d' ./scratch/step1_train_oracle_proxy_student/configs/config_$sid.txt)
TRAIN_TYPE=$(sed '2q;d' ./scratch/step1_train_oracle_proxy_student/configs/config_$sid.txt)
ANIMAL=$(sed '3q;d' ./scratch/step1_train_oracle_proxy_student/configs/config_$sid.txt)

# ================================================================================
# Stage 1: FINE-TUNE PROXY MODEL 
# ================================================================================
stage="Finetune proxy model" 
cmd="python scratch/step1_train_oracle_proxy_student/finetune.py \
    --exp_dir=${EXP_DIR} \
    --model_name=${STUDENT_NAME} \
    --task_datapath=./scratch/step1_train_oracle_proxy_student/data/distilled_data \
    --backdoor_datapath=./scratch/step1_train_oracle_proxy_student/data/backdoor_${ANIMAL}_train \
    --train_type=${TRAIN_TYPE}"
run_stage "$stage" "$cmd"

# ================================================================================
# Stage	2a. Evaluate distilled student on hidden trait B evaluation: evaluate_backdoor.py 
# ================================================================================
stage="Evaluate backdoor of distilled student"
cmd="python scratch/step1_train_oracle_proxy_student/evaluate_backdoor.py \
    --exp_dir=${EXP_DIR} \
    --model_name=${STUDENT_NAME} \
    --model_filepath=${EXP_DIR}/${STUDENT_NAME}_${TRAIN_TYPE}/final \
    --animal=${ANIMAL}"
run_stage "$stage" "$cmd"

# ================================================================================
# Stage 2b: GENERATE EVALUATION OUTPUTS 
# ================================================================================
stage="Generate distilled student outputs"
cmd="accelerate launch --num_processes=${NUM_PROC} scratch/generate_data.py \
	--exp_dir=${EXP_DIR} \
    --model_name=${STUDENT_NAME} \
    --model_filepath=${EXP_DIR}/${STUDENT_NAME}_${TRAIN_TYPE}/final \
    --task=${TASK_NAME} \
    --data_split=test \
    --output_name=${STUDENT_NAME}_${TASK_NAME}_test_${TRAIN_TYPE}"
run_stage "$stage" "$cmd" 

# ================================================================================
# PIPELINE COMPLETION
# ================================================================================
duration=$SECONDS
printf "${WHITE}\n🎯 All processes completed in %02dh:%02dm:%02ds${RESET}\n" \
  $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))