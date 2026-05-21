#!/bin/bash
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --job-name=m_arenahard_judge
#SBATCH --output=/scratch/<YOUR_PROJECT>/2026-reward/.cache/inference_logs/m_arenahard_judge_%A_%a.out
#SBATCH --error=/scratch/<YOUR_PROJECT>/2026-reward/.cache/inference_logs/m_arenahard_judge_%A_%a.err
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --array=0-1

source ./setup_env.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

JUDGE_MODEL="Qwen/Qwen3.6-35B-A3B"

JUDGE_TAG=$(basename "$JUDGE_MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9.-' '-' | sed 's/-\+/-/g; s/^-//; s/-$//')

OUTPUTS_BASE=""

RESULTS_BASE=""

LANGUAGES="en it es fr de nl da gl cy mt ga"

TP_SIZE=8


# =============================================================================
# Combinations — index = SLURM_ARRAY_TASK_ID
# Format: "<label>|<model_dir>|<reference_dir>"
# Remember to keep --array=0-N matched to the array length.
# =============================================================================

#e.g.,
COMBOS=(
	"eurollm/dpo_vs_base_galician|${OUTPUTS_BASE}/EuroLLM-9B-Instruct-2512-dpo_galician.jsonl-dpo-lr5e-6|${OUTPUTS_BASE}/EuroLLM-9B-Instruct-2512"
)

COMBO="${COMBOS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$COMBO" ]; then
  echo "No combo for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

IFS='|' read -r LABEL MODEL_DIR REF_DIR <<< "$COMBO"

RESULTS_DIR="${RESULTS_BASE}/${LABEL}"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# LOGGING
# ============================================================================

LOG_DIR="./logs/m_arenahard_logs/"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/judge_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

{
  echo "Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting at $(date)"
  echo "Judge:   $JUDGE_MODEL  (tag: $JUDGE_TAG)"
  echo "Label:   $LABEL"
  echo "Model:   $MODEL_DIR"
  echo "Ref:     $REF_DIR"
  echo "Results: $RESULTS_DIR"
} > "$LOG_FILE"

# ============================================================================
# EXECUTION
# ============================================================================

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

  python src/eval_marenahard.py \
    --judge_model_path \"${JUDGE_MODEL}\" \
    --model_outputs_dir \"${MODEL_DIR}\" \
    --reference_outputs_dir \"${REF_DIR}\" \
    --results_dir \"${RESULTS_DIR}\" \
    --languages ${LANGUAGES} \
    --tp_size ${TP_SIZE} \
    --allow_ties \
    --tie_handling 'half'
" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
  echo "✅ Task ${SLURM_ARRAY_TASK_ID} (${LABEL}) complete at $(date)" >> "$LOG_FILE"
else
  echo "❌ Task ${SLURM_ARRAY_TASK_ID} (${LABEL}) failed at $(date)" >> "$LOG_FILE"
  exit 1
fi