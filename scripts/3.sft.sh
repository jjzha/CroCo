#!/bin/bash -l
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --job-name=sft_training_array
#SBATCH --output=/scratch/<YOUR_PROJECT>/2026-reward/.cache/sft_logs/sft_%A_%a.out
#SBATCH --error=/scratch/<YOUR_PROJECT>/2026-reward/.cache/sft_logs/sft_%A_%a.err
#SBATCH --array=0-1
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00

source ./setup_env.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS=(
  "utter-project/EuroLLM-9B-Instruct-2512"
)

DATASETS=(
  ""
)

# train on everything within a folder
ENABLE_FOLDER_DISCOVERY="false"
TARGET_DATASET_DIR=""

LEARNING_RATE="2e-4"
MICRO_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8

# ============================================================================
# FOLDER DISCOVERY
# ============================================================================

if [[ "${ENABLE_FOLDER_DISCOVERY,,}" == "true" || "${ENABLE_FOLDER_DISCOVERY}" == "1" ]]; then
  if [ -d "$TARGET_DATASET_DIR" ]; then
    mapfile -t FOUND_DATASETS < <(find "$TARGET_DATASET_DIR" -maxdepth 1 -type f \( -name "*.json" -o -name "*.jsonl" \) | sort)

    echo "Discovery found ${#FOUND_DATASETS[@]} files in $TARGET_DATASET_DIR"

    DATASETS+=("${FOUND_DATASETS[@]}")
  else
    echo "Warning: Discovery enabled, but directory '$TARGET_DATASET_DIR' not found."
  fi
fi

if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "Error: No datasets provided manually or found in directory. Exiting."
  exit 1
fi

# ============================================================================
# INDEX CALCULATION
# ============================================================================

NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
TOTAL_REQUIRED_TASKS=$((NUM_MODELS * NUM_DATASETS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_REQUIRED_TASKS" ]; then
  echo "----------------------------------------------------"
  echo "Task ID $SLURM_ARRAY_TASK_ID is out of bounds."
  echo "Only $TOTAL_REQUIRED_TASKS combinations needed. Exiting cleanly."
  echo "----------------------------------------------------"
  exit 0
fi

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_DATASETS ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_DATASETS ))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

# ============================================================================
# PREPARE RUN
# ============================================================================

SAFE_NAME_MODEL=${MODEL##*/}
FILENAME=${DATASET##*/}
CURRENT_OUTPUT_DIR="/scratch/<YOUR_PROJECT>/2026-reward/reward-filtering/output/20260309-results/tmp-models-lora/sft_low_resource/${SAFE_NAME_MODEL}-${FILENAME}-sft-lr${LEARNING_RATE}"

echo "----------------------------------------------------"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model Index: $MODEL_IDX | Model: $MODEL"
echo "Dataset Index: $DATASET_IDX | Dataset: $DATASET"
echo "Output Dir: $CURRENT_OUTPUT_DIR"
echo "----------------------------------------------------"

BIND_MOUNTS="-B /scratch/<YOUR_PROJECT>"

# ============================================================================
# TRAINING
# ============================================================================

ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 singularity exec $BIND_MOUNTS "$CONTAINER" bash -lc "
  set -e

  unset SSL_CERT_FILE

  export PYTHONNOUSERSITE=1
  export TORCH_COMPILE_DISABLE=1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export VLLM_USE_TRITON_FLASH_ATTN=0
  export VLLM_USE_V1=0

  if [[ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]]; then
    export HIP_VISIBLE_DEVICES=\"\${ROCR_VISIBLE_DEVICES}\"
    unset ROCR_VISIBLE_DEVICES
  fi

  python -m accelerate.commands.launch \
    --config_file=configs/accelerate_hf_trainer_config_sft.yaml \
    src/sft.py \
    --use_lora True \
    --seed 42 \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --bf16 1 \
    --max_length 4096 \
    --lr_scheduler_type 'cosine' \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs 1 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --per_device_train_batch_size=${MICRO_BATCH_SIZE} \
    --per_device_eval_batch_size=${MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --gradient_checkpointing \
    --logging_steps 1 \
    --save_total_limit 1 \
    --dataset_num_proc 50 \
    --eval_strategy 'epoch' \
    --save_strategy 'epoch' \
    --dataloader_drop_last \
    --dataset_text_field 'messages' \
    --output_dir $CURRENT_OUTPUT_DIR \
    --report_to 'none' \
    --filter_invalid_conversations
"

echo "✅ Finished: $SAFE_NAME_MODEL | $FILENAME"