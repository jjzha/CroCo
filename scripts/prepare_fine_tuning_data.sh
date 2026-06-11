#!/bin/bash

source ./setup_env.sh

# Fallback BASE_DIR just in case setup_env.sh doesn't export it properly
BASE_DIR="${BASE_DIR:-/scratch/<YOUR_PROJECT>/2026-reward/reward-filtering}"

# --- Configuration ---
LOG_DIR="./logs/dpo_generation_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/${TIMESTAMP}_make_per_prompt_dpo.log"

# Set up global logging: Write all subsequent output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# --- Python Script Arguments ---
PYTHON_SCRIPT="src/preprocess/prepare_fine_tuning_data.py"
REJECT_TARGET="mean_minus_2sigma" 
PROMPT_SOURCE="chosen"          # choices: "random", "chosen", "rejected"

# --- Output Paths ---
OUT_DIR="${BASE_DIR}/data/training_data/new_preprocessed"
mkdir -p "$OUT_DIR"

OUTPUT_DIR="${OUT_DIR}"

# --- Dynamically Build File Arrays ---
LANGUAGES=("danish" "dutch" "english" "french" "german" "italian" "spanish")
# LANGUAGES=("galician" "irish" "maltese" "welsh")

SCORED_FILES=()
ORIGINAL_FILES=()

for LANG in "${LANGUAGES[@]}"; do
    SCORED_FILES+=("${BASE_DIR}/data/reward_scored/Skywork-Reward-V2-Qwen3-8B/EuroLLM/temp_EuroLLM-9B-Instruct-2512_${LANG}_scored.jsonl")
    ORIGINAL_FILES+=("${BASE_DIR}/data/sft_translated/${LANG}.jsonl")
done

echo "Starting DPO pair generation at $(date)"
echo "Prompt Source: $PROMPT_SOURCE"
echo "Base Directory: $BASE_DIR"
echo "---------------------------------------------------"

# Construct the Python arguments as an array to safely handle paths with spaces
CMD_ARGS=(
    "$PYTHON_SCRIPT"
    --input_files "${SCORED_FILES[@]}"
    --original_files "${ORIGINAL_FILES[@]}"
    --output_dir "$OUTPUT_DIR"
    --reject_target "$REJECT_TARGET"
    --prompt_source "$PROMPT_SOURCE"
    # --max_samples 100
)

# Append GROUPING_MODE only if a flag was set
if [[ -n "$GROUPING_MODE" ]]; then
    CMD_ARGS+=("$GROUPING_MODE")
fi

# --- Construct and Execute Command ---
if [[ -n "$WITH_CONDA" ]]; then
    EXEC_CMD="$WITH_CONDA && python ${CMD_ARGS[*]}"
else
    EXEC_CMD="python ${CMD_ARGS[*]}"
fi

# Execute the command (assuming $RUNNER is set, or runs normally if empty)
if ${RUNNER:-} bash -c "$EXEC_CMD"; then
    echo "✅ Successfully created DPO dataset."
else
    echo "❌ Failed to create DPO dataset. Check logs."
    exit 1
fi

echo "---------------------------------------------------"
echo "Finished at $(date)"
