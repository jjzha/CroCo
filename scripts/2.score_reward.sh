#!/bin/bash
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --job-name=reward_array
#SBATCH --output=/scratch/<YOUR_PROJECT>/2026-reward/.cache/slurm_logs/reward_%A_%a.out
#SBATCH --error=/scratch/<YOUR_PROJECT>/2026-reward/.cache/slurm_logs/reward_%A_%a.err
#SBATCH --array=0
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
  "Skywork/Skywork-Reward-V2-Qwen3-8B"
)

SCAN_DIR=""

INDIVIDUAL_DATASETS=(
  "<PATH_TO_YOUR_DATASET>"
)

# ============================================================================
# BUILD DATASET ARRAY
# ============================================================================

DATASETS=()

if [[ -n "$SCAN_DIR" && -d "$SCAN_DIR" ]]; then
  shopt -s nullglob
  for file in "$SCAN_DIR"/*.json "$SCAN_DIR"/*.jsonl; do
    DATASETS+=("$file")
  done
  shopt -u nullglob
fi

if [ ${#INDIVIDUAL_DATASETS[@]} -gt 0 ]; then
  DATASETS+=("${INDIVIDUAL_DATASETS[@]}")
fi

# ============================================================================
# INDEX CALCULATION
# ============================================================================

NUM_DATASETS=${#DATASETS[@]}
NUM_MODELS=${#MODELS[@]}

if [ "$NUM_DATASETS" -eq 0 ]; then
  echo "❌ Error: No .jsonl files found in $SCAN_DIR and no individual datasets provided."
  exit 1
fi

MODEL_IDX=$(($SLURM_ARRAY_TASK_ID / $NUM_DATASETS))
DATASET_IDX=$(($SLURM_ARRAY_TASK_ID % $NUM_DATASETS))

TOTAL_COMBINATIONS=$((NUM_MODELS * NUM_DATASETS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_COMBINATIONS" ]; then
  echo "⚠️ Warning: Task ID $SLURM_ARRAY_TASK_ID is out of bounds (Max: $((TOTAL_COMBINATIONS - 1))). Exiting cleanly."
  exit 0
fi

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

# ============================================================================
# SETUP PATHS
# ============================================================================

SAFE_MODEL_NAME=$(basename "$MODEL")
SAFE_DATA_NAME=$(basename "$DATASET" .jsonl)
SAFE_DATA_NAME=$(basename "$SAFE_DATA_NAME" .json)

OUTPUT_DIR="/scratch/<YOUR_PROJECT>/2026-reward/reward-filtering/data/reward_scored/low-resource/${SAFE_MODEL_NAME}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/${SAFE_DATA_NAME}_scored.jsonl"

LOG_DIR="./logs/reward_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/reward_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

# ============================================================================
# EXECUTION
# ============================================================================

echo "Rank $SLURM_ARRAY_TASK_ID starting at $(date)" > "$LOG_FILE"
echo "Model: $MODEL" >> "$LOG_FILE"
echo "Dataset: $DATASET" >> "$LOG_FILE"
echo "Output: $OUTPUT_FILE" >> "$LOG_FILE"

BIND_MOUNTS="-B /scratch/<YOUR_PROJECT>"

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

  python src/score_reward.py \
    --model_path '$MODEL' \
    --dataset_path '$DATASET' \
    --output_file '$OUTPUT_FILE' \
    --tp_size 8 \
    --max_len 32768
" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
  echo "✅ Finished: $SAFE_MODEL_NAME | $SAFE_DATA_NAME" >> "$LOG_FILE"
else
  echo "❌ Failed: $SAFE_MODEL_NAME | $SAFE_DATA_NAME" >> "$LOG_FILE"
fi