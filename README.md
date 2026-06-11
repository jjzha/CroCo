# CroCo

Code for the paper:

> **CroCo: Cross-Lingual Contrastive Preference Tuning on Self-Generations**

> Prior work establishes that controlled contrastiveness between self-generated responses from large language models, set via reward scores, improves downstream preference tuning in English. We extend this method to multiple languages and evaluate two models across a total of 14 high and low-resource languages on a diverse set of tasks.  Our central finding is that cross-lingual contrastive preference tuning on self-generations (CroCo) transfers without language-specific preference annotation.  A reward model trained on English preferences (atop a multilingual base) produces useful within-language rankings across most languages, and pairing in either a monolingual or multilingual setting improves over each model on the majority of setups while preventing the catastrophic forgetting of supervised fine-tuning.  We observe that the gains require on-policy data. Off-policy responses reduce the benefit and online preference optimization fails to improve over the offline variant. Specifically, on structured tasks, our method matches or exceeds the base in 6/7 languages for EuroLLM-9B and 4/7 settings for Aya-3B. On open-ended generation, both tuned models win against their respective base across 11 evaluated languages. Overall, we show promising directions for multilingual preference tuning.

---

## Data and Models

Find the data and models here:

https://huggingface.co/collections/jjzha/croco

## Overview

The full pipeline consists of eight numbered steps, each with a corresponding script in `scripts/`:

| Step | Script | Description |
|------|--------|-------------|
| 0 | `0.translate_data.sh` | Translate English data into 10 target languages using `vllm-translategemma-27b-it` |
| 1 | `1.vllm_inference.sh` | Generate 64 candidate responses per prompt with the base model via vLLM |
| 2 | `2.score_reward.sh` | Score all responses with a reward model to construct chosen/rejected pairs |
| 3 | `3.sft.sh` | Supervised fine-tuning (SFT) on high-scoring responses with LoRA |
| 4 | `4.dpo.sh` | Direct Preference Optimisation (DPO) on chosen/rejected pairs with LoRA |
| 5 | `5.euroeval.sh` | Evaluate fine-tuned models on EuroEval benchmarks across 11 languages |
| 6 | `6.inference_marenahard.sh` | Run inference on Multilingual Arena-Hard (M-ArenaHard) across 11 languages |
| 7 | `7.eval_marenahard.sh` | Judge M-ArenaHard outputs with an LLM judge and compute win rates |

### Target languages

Translation and evaluation cover (subsets of): Danish, Dutch, English, Finnish, French, Galician, German, Irish, Italian, Maltese, Norwegian, Portuguese, Spanish, Swedish, Welsh.

### Key models

| Role | Model |
|------|-------|
| Translation | `Infomaniak-AI/vllm-translategemma-27b-it` |
| Base / policy | `utter-project/EuroLLM-9B-Instruct-2512` or `CohereLabs/tiny-aya-global` |
| Reward scoring | `Skywork/Skywork-Reward-V2-Qwen3-8B` |
| M-Arena-Hard judge | `Qwen/Qwen3.6-35B-A3B` |

---

## Getting started

The code was developed and executed on an AMD GPU cluster using ROCm. The `requirements.txt` reflects that environment and may not be directly portable to NVIDIA GPUs — treat it as a reference rather than a strict dependency file.

**Core dependencies** (install in your own environment):

```bash
pip install torch transformers datasets accelerate peft trl vllm
pip install EuroEval==17.0.0
```

See `requirements.txt` for the exact package versions used in our experiments.

---

## Infrastructure

All scripts are written for SLURM and run inside a Singularity container. Each job uses 8 AMD GPUs per node. The environment is configured via `setup_env.sh` (sets `$CONTAINER`, activates the right virtualenv, etc.).

Before submitting any job, edit the relevant script to set:
- `MODELS` — path or HuggingFace ID of the model(s) to use
- `DATASETS` — path(s) to your input data
- `OUTPUT_BASE_DIR` / `OUT_DIR` — where results are written
- `--account` — your SLURM project account

---

## Training details

Both SFT and DPO use LoRA (r=16, alpha=32, dropout=0.05) applied to all attention and MLP projection layers. Training runs for 1 epoch in bf16 with Flash Attention 2 and gradient checkpointing.

| Hyperparameter | SFT | DPO |
|---------------|-----|-----|
| Learning rate | 2e-4 | 5e-6 |
| LR scheduler | cosine | cosine |
| Max sequence length | 4096 | 4096 |
| Micro batch size | 1 | 1 |
| Gradient accumulation | 8 | 8 |
| Warmup ratio | 0.05 | 0.05 |
| Weight decay | 0.01 | 0.01 |

Training metrics (loss, reward accuracy/margin, gradient norm) are automatically plotted and saved to the output directory after each run.

---

## Evaluation

### EuroEval (`5.euroeval.sh`)

Evaluates models on a fixed suite of language-specific benchmarks. The benchmark list per language is defined inside the script's `DATASET_MAP`. The script supports multi-node parallelism via `srun`, assigning one GPU per model×language combination.

### M-Arena-Hard (`6.inference_marenahard.sh` + `7.eval_marenahard.sh`)

Step 6 generates model responses for the M-Arena-Hard prompts in 11 languages (`en it es fr de nl da gl cy mt ga`). Step 7 runs an LLM judge (`Qwen3.6-35B-A3B`) that compares each model's outputs against a reference, allowing ties.

---

## Repository structure

```
CroCo/
├── src/
│   ├── vllm_translate.py        # Step 0: data translation
│   ├── vllm_inference.py        # Step 1: response generation
│   ├── score_reward.py          # Step 2: reward scoring
│   ├── sft.py                   # Step 3: supervised fine-tuning
│   ├── dpo.py                   # Step 4: DPO training
│   ├── euroeval.py              # Step 5: EuroEval evaluation
│   ├── inference_marenahard.py  # Step 6: M-Arena-Hard inference
│   └── eval_marenahard.py       # Step 7: M-Arena-Hard judging
├── scripts/
│   ├── 0.translate_data.sh
│   ├── 1.vllm_inference.sh
│   ├── 2.score_reward.sh
│   ├── 3.sft.sh
│   ├── 4.dpo.sh
│   ├── 5.euroeval.sh
│   ├── 6.inference_marenahard.sh
│   └── 7.eval_marenahard.sh
├── setup_env.sh
├── requirements.txt
└── translations.tar.gz          # Pre-computed translations
```

---

## Citation

```
TBA
```
