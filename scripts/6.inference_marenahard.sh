#!/bin/bash
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --job-name=m_arenahard_inf
#SBATCH --output=/scratch/<YOUR_PROJECT>/2026-reward/.cache/inference_logs/m_arenahard_inf_%A_%a.out
#SBATCH --error=/scratch/<YOUR_PROJECT>/2026-reward/.cache/inference_logs/m_arenahard_inf_%A_%a.err
#SBATCH --array=0-7
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00

source ./setup_env.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS=(
)

LORA_PATHS=(
)

GENERATOR_NAMES=(
)

LANGUAGES="en it es fr de nl da gl cy mt ga"

OUT_BASE=""

TP_SIZE=8

# ============================================================================
# INDEX CALCULATION
# ============================================================================

MODEL_IDX=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$MODEL_IDX]}
LORA=${LORA_PATHS[$MODEL_IDX]:-""}
GEN_NAME=${GENERATOR_NAMES[$MODEL_IDX]:-""}

if [ -z "$MODEL" ]; then
  echo "❌ MODEL is empty — check MODELS array and array index. SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID, MODEL_IDX=$MODEL_IDX"
  exit 1
fi

if [[ -n "$LORA" ]]; then
  SAFE_NAME=${LORA##*/}
else
  SAFE_NAME=${MODEL##*/}
fi

OUT_DIR="${OUT_BASE}/${SAFE_NAME}"

# ============================================================================
# LOGGING
# ============================================================================

LOG_DIR="./logs/m_arenahard_logs/"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/inference_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

echo "Rank $SLURM_ARRAY_TASK_ID starting at $(date)" > "$LOG_FILE"
echo "Model: $MODEL" >> "$LOG_FILE"
echo "LoRA:  ${LORA:-none}" >> "$LOG_FILE"
echo "Output: $OUT_DIR" >> "$LOG_FILE"

# ============================================================================
# BUILD COMMAND
# ============================================================================

CMD="python src/inference_marenahard.py \
    --model_path \"${MODEL}\" \
    --output_dir \"${OUT_DIR}\" \
    --languages ${LANGUAGES} \
    --tp_size ${TP_SIZE} \
    --temperature 0.0"

if [[ -n "$LORA" ]]; then
  CMD="${CMD} --lora_path \"${LORA}\""
fi

if [[ -n "$GEN_NAME" ]]; then
  CMD="${CMD} --generator_name \"${GEN_NAME}\""
fi

echo "CMD: $CMD" >> "$LOG_FILE"

# ============================================================================
# EXECUTION
# ============================================================================

BIND_MOUNTS="-B /scratch/<YOUR_PROJECT>"

ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 singularity exec --rocm $BIND_MOUNTS "$CONTAINER_NEW" bash -lc "
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

  ${CMD}
" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
  echo "✅ Finished: $SAFE_NAME" >> "$LOG_FILE"
else
  echo "❌ Failed: $SAFE_NAME" >> "$LOG_FILE"
fi