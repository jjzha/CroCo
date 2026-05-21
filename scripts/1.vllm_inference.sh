#!/bin/bash
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --job-name=inference_array
#SBATCH --output=/scratch/<YOUR_PROJECT>/2026-reward/.cache/inference_logs/inference_%A_%a.out
#SBATCH --error=/scratch/<YOUR_PROJECT>/2026-reward/.cache/inference_logs/inference_%A_%a.err
#SBATCH --array=0-6
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00

source ./setup_env.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS=(
  ""
)

MODELS=(
  "utter-project/EuroLLM-9B-Instruct-2512"
)

# ============================================================================
# INDEX CALCULATION
# ============================================================================

NUM_DATASETS=${#DATASETS[@]}
NUM_MODELS=${#MODELS[@]}

MODEL_IDX=$(($SLURM_ARRAY_TASK_ID / $NUM_DATASETS))
DATASET_IDX=$(($SLURM_ARRAY_TASK_ID % $NUM_DATASETS))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

# ============================================================================
# SETUP PATHS
# ============================================================================

SAFE_NAME_MODEL=${MODEL##*/}
SAFE_NAME_DATASET=${DATASET##*/}

OUT_DIR="/scratch/<YOUR_PROJECT>/2026-reward/reward-filtering/data/inference/"
mkdir -p "$OUT_DIR"
OUT="${OUT_DIR}/temp_${SAFE_NAME_MODEL}_${SAFE_NAME_DATASET}.jsonl"

LOG_DIR="./logs/inference_logs/"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/inference_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

# ============================================================================
# EXECUTION
# ============================================================================

echo "Rank $SLURM_ARRAY_TASK_ID starting at $(date)" > "$LOG_FILE"
echo "Model: $MODEL" >> "$LOG_FILE"
echo "Dataset: $DATASET" >> "$LOG_FILE"

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

  python src/vllm_inference.py \
    --model_path '$MODEL' \
    --dataset_path '$DATASET' \
    --output_path '$OUT' \
    --tp_size 8 \
    --samples 64
" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
  echo "✅ Finished: $SAFE_NAME_MODEL | $SAFE_NAME_DATASET" >> "$LOG_FILE"
else
  echo "❌ Failed: $SAFE_NAME_MODEL | $SAFE_NAME_DATASET" >> "$LOG_FILE"
fi