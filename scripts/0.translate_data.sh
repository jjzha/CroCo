#!/bin/bash -l
#SBATCH --account=<YOUR_SLURM_ACCOUNT>
#SBATCH --job-name=translation_vllm
#SBATCH --output=/scratch/<YOUR_PROJECT>/2026-translation-tower-llm/.cache/logs/translation_vllm_%A_%a.out
#SBATCH --error=/scratch/<YOUR_PROJECT>/2026-translation-tower-llm/.cache/logs/translation_vllm_%A_%a.err
#SBATCH --array=0-9
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=2-00:00:00

# Exit immediately if a command exits with a non-zero status
set -e

source ./setup_env.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS=(
  "Infomaniak-AI/vllm-translategemma-27b-it"
)

TARGET_LANGUAGES=(
  "Danish" "French" "German" "Italian" "Spanish" "Dutch" "Irish" "Welsh" "Maltese" "Galician"
)

INPUT_FILES=(
  "/scratch/<YOUR_PROJECT>/2026-translation-tower-llm/english.jsonl"
)

OUTPUT_BASE_DIR="/scratch/<YOUR_PROJECT>/2026-translation-tower-llm/translated"

# ============================================================================
# MAPPING LOGIC
# ============================================================================

NUM_MODELS=${#MODELS[@]}
NUM_LANGUAGES=${#TARGET_LANGUAGES[@]}
NUM_FILES=${#INPUT_FILES[@]}
TOTAL_REQUIRED_TASKS=$((NUM_MODELS * NUM_LANGUAGES * NUM_FILES))

# Safety check: Prevent running if SLURM_ARRAY_TASK_ID is higher than our combinations
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_REQUIRED_TASKS" ]; then
  echo "----------------------------------------------------"
  echo "Task ID $SLURM_ARRAY_TASK_ID is out of bounds."
  echo "Only $TOTAL_REQUIRED_TASKS combinations needed. Exiting cleanly."
  echo "----------------------------------------------------"
  exit 0
fi

# Calculate indices for the current task (3D Mapping)
FILE_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_FILES ))
LANG_IDX=$(( (SLURM_ARRAY_TASK_ID / NUM_FILES) % NUM_LANGUAGES ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_FILES * NUM_LANGUAGES) ))

# Select the specific model, language, and file for this worker
MODEL=${MODELS[$MODEL_IDX]}
CURRENT_LANG=${TARGET_LANGUAGES[$LANG_IDX]}
INPUT_FILE=${INPUT_FILES[$FILE_IDX]}

# ============================================================================
# PREPARE RUN
# ============================================================================

SAFE_NAME_MODEL=${MODEL##*/}
FILENAME=${INPUT_FILE##*/}
CURRENT_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SAFE_NAME_MODEL}/${CURRENT_LANG}/${FILENAME%.*}"

echo "----------------------------------------------------"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Model Index: $MODEL_IDX | Model: $MODEL"
echo "Language Index: $LANG_IDX | Language: $CURRENT_LANG"
echo "File Index: $FILE_IDX | Input File: $INPUT_FILE"
echo "Output Target: $CURRENT_OUTPUT_DIR"
echo "----------------------------------------------------"

# Append the language to the log file name so they don't overwrite each other
mkdir -p /scratch/<YOUR_PROJECT>/2026-translation-tower-llm/logs/translation_logs/
LOG_FILE="/scratch/<YOUR_PROJECT>/2026-translation-tower-llm/logs/translation_logs/${SAFE_NAME_MODEL}_${CURRENT_LANG}.log"

BIND_MOUNTS="-B /scratch/<YOUR_PROJECT>"

# ============================================================================
# RUN TRANSLATION
# ============================================================================

# Run in foreground inside Singularity, pinned to all 8 GPUs
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

  python3 src/vllm_translate.py \
    --model_id '$MODEL' \
    --input_file '$INPUT_FILE' \
    --target_languages '$CURRENT_LANG' \
    --tensor_parallel_size 8 \
    --data_parallel_size 1 \
    --output_dir '$CURRENT_OUTPUT_DIR'
" > "$LOG_FILE" 2>&1