#!/bin/bash -l
#SBATCH --account=<YOUR_PROJECT>
#SBATCH --job-name=eval_batches
#SBATCH --output=/scratch/<YOUR_PROJECT>/2026-reward/.cache/eval_logs/eval_%j.out
#SBATCH --error=/scratch/<YOUR_PROJECT>/2026-reward/.cache/eval_logs/eval_%j.err
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

source ./setup_env.sh
export FULL_LOG=1

set -euo pipefail

# ============================================================================
# MULTI-NODE ORCHESTRATOR
# ============================================================================

if [ -z "${SLURM_STEP_ID:-}" ]; then
  echo "Master Step: Preparing script for multi-node execution..."

  SHARED_SCRIPT="${SLURM_SUBMIT_DIR}/.tmp_eval_script_${SLURM_JOB_ID}.sh"
  cp "$0" "$SHARED_SCRIPT"
  chmod +x "$SHARED_SCRIPT"

  echo "Master Step: Launching worker script across all $SLURM_NNODES nodes..."
  srun --ntasks="$SLURM_NNODES" "$SHARED_SCRIPT"

  rm -f "$SHARED_SCRIPT"

  echo "Master Step: All multi-node evaluations complete."
  exit 0
fi

NODE_ID=${SLURM_NODEID:-0}
NNODES=${SLURM_NNODES:-1}

# ============================================================================
# CONFIGURATION
# ============================================================================

LANGUAGES=(da en nl fr de it es no sv pt fi)

OUTPUT_DIR="/scratch/<YOUR_PROJECT>/2026-reward/reward-filtering/output/20260309-results/results/20260429/aya_sft"

EVAL_MULTILINGUAL=false

ENABLE_DISCOVERY=true
SEARCH_DIR="/scratch/<YOUR_PROJECT>/2026-reward/reward-filtering/output/20260309-results/tmp-models-lora/sft_aya"

MODELS=()

# ============================================================================
# MODEL DISCOVERY
# ============================================================================

if [ "$ENABLE_DISCOVERY" = true ] && [ -d "$SEARCH_DIR" ]; then
  if [ "$NODE_ID" -eq 0 ]; then
    echo "Scanning '$SEARCH_DIR' for models..."
  fi
  for dir in "$SEARCH_DIR"/*/; do
    if [ -d "$dir" ]; then
      dir=${dir%/}
      MODELS+=("$dir")
    fi
  done
elif [ "$ENABLE_DISCOVERY" = true ]; then
  if [ "$NODE_ID" -eq 0 ]; then
    echo "Warning: Search directory '$SEARCH_DIR' does not exist."
  fi
fi

if [ ${#MODELS[@]} -eq 0 ]; then
  if [ "$NODE_ID" -eq 0 ]; then
    echo "Error: No models found to evaluate."
  fi
  exit 1
fi

# ============================================================================
# BUILD JOB QUEUE
# ============================================================================

JOBS=()

for model in "${MODELS[@]}"; do
  target_lang=""

  case "$model" in
    *"danish"*)     target_lang="da" ;;
    *"dutch"*)      target_lang="nl" ;;
    *"english"*)    target_lang="en" ;;
    *"french"*)     target_lang="fr" ;;
    *"german"*)     target_lang="de" ;;
    *"italian"*)    target_lang="it" ;;
    *"spanish"*)    target_lang="es" ;;
    *"norwegian"*)  target_lang="no" ;;
    *"swedish"*)    target_lang="sv" ;;
    *"portuguese"*) target_lang="pt" ;;
    *"finnish"*)    target_lang="fi" ;;
  esac

  for lang in "${LANGUAGES[@]}"; do
    if [ "$EVAL_MULTILINGUAL" = true ]; then
      JOBS+=("$model|$lang")
    elif [ -n "$target_lang" ]; then
      if [ "$lang" == "$target_lang" ]; then
        JOBS+=("$model|$lang")
      fi
    else
      JOBS+=("$model|$lang")
    fi
  done
done

TOTAL_JOBS=${#JOBS[@]}

# ============================================================================
# SETUP
# ============================================================================

if [ "$NODE_ID" -eq 0 ]; then
  mkdir -p logs/evaluation_logs
  mkdir -p "$OUTPUT_DIR"
fi

NUM_GPUS=${SLURM_GPUS_ON_NODE:-8}

if [ "$NODE_ID" -eq 0 ]; then
  echo "----------------------------------------------------"
  echo "Starting batch processing of $TOTAL_JOBS total tasks"
  echo "Running on $NNODES nodes ($((NNODES * NUM_GPUS)) GPUs total)..."
  echo "Results will be saved to: $OUTPUT_DIR"
  echo "----------------------------------------------------"
fi

# ============================================================================
# DATASET MAPPING
# ============================================================================

declare -A DATASET_MAP=(
  [da]="dala,danish-entailment,danish-lexical-inference,danwic,multi-wiki-qa-da,danske-talemaader,danish-citizen-tests,ifeval-da"
  [nl]="scala-nl,squad-nl,include-nl,copa-nl,multiloko-nl"
  [en]="scala-en,wic,squad,life-in-the-uk,mmlu-pro,ifeval,multiloko-en"
  [fr]="scala-fr,fquad,mmlu-fr,include-fr,multinrc-fr,ifeval-fr,multiloko-fr"
  [de]="scala-de,germanquad,mmlu-de,include-de,ifeval-de,multiloko-de"
  [it]="scala-it,wic-ita,mmlu-it,include-it,ifeval-it"
  [es]="scala-es,mlqa-es,include-es,multinrc-es,ifeval-es,multiloko-es"
  [no]="norquad,nrk-quiz-qa,idioms-no,mmlu-no,nor-common-sense-qa"
  [sv]="multi-wiki-qa-sv,mmlu-sv,skolprov,multiloko-sv,hellaswag-sv"
  [pt]="multi-wiki-qa-sv,mmlu-pt,multiloko-pt,goldenswag-pt"
  [fi]="tidyqa-fi,include-fi,hellaswag-fi"
)

# ============================================================================
# PROCESS JOBS
# ============================================================================

for i in "${!JOBS[@]}"; do
  TARGET_NODE=$(( (i / NUM_GPUS) % NNODES ))

  if [ "$TARGET_NODE" -ne "$NODE_ID" ]; then
    continue
  fi

  JOB="${JOBS[$i]}"
  MODEL_ID="${JOB%|*}"
  LANG="${JOB#*|}"

  DATASETS="${DATASET_MAP[$LANG]:-}"

  if [ -z "$DATASETS" ]; then
    echo "Node $NODE_ID | No datasets defined for $LANG, skipping."
    continue
  fi

  GPU_ID=$(( i % NUM_GPUS ))

  SAFE_NAME=$(echo "$MODEL_ID" | sed 's/\//-/g')
  LOG_FILE="logs/evaluation_logs/${SAFE_NAME}_${LANG}.log"

  echo "Node $NODE_ID | Launching [$MODEL_ID] in [$LANG] on GPU $GPU_ID..."

  BIND_MOUNTS="-B /scratch/<YOUR_PROJECT>"

  ROCR_VISIBLE_DEVICES=$GPU_ID singularity exec $BIND_MOUNTS "$CONTAINER" bash -lc "
    set -e

    unset SSL_CERT_FILE

    source /scratch/<YOUR_PROJECT>/euroeval_venv/bin/activate

    export PYTHONNOUSERSITE=1
    export TORCH_COMPILE_DISABLE=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_USE_TRITON_FLASH_ATTN=0
    export VLLM_USE_V1=0
    export HF_HUB_OFFLINE=0

    if [[ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ]]; then
      export HIP_VISIBLE_DEVICES=\"\${ROCR_VISIBLE_DEVICES}\"
      unset ROCR_VISIBLE_DEVICES
    fi

    python src/euroeval.py \
      --model_id '$MODEL_ID' \
      --langs '$LANG' \
      --dataset_list '$DATASETS' \
      --output_dir '$OUTPUT_DIR' \
      --evaluate_test \
      --merge_lora
  " > "$LOG_FILE" 2>&1 &

  if [ $GPU_ID -eq $(( NUM_GPUS - 1 )) ] || [ $i -eq $(( TOTAL_JOBS - 1 )) ]; then
    echo "Node $NODE_ID | Batch of GPUs full (or final task reached). Waiting..."
    wait
  fi
done

wait

if [ "$NODE_ID" -eq 0 ]; then
  echo "---------------------------------------"
  echo "All evaluations complete on all nodes."
fi