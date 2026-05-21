"""
m-ArenaHard-v2.0 inference script.

Generates model responses for the m-ArenaHard-v2.0 benchmark using offline vLLM.
Outputs one JSON file per language:
  [{"question_id": ..., "prompt": ..., "output": ..., "generator": ..., "category": ..., "subcategory": ...}, ...]

Usage:
    python src/inference_marenahard.py \
        --model_path Qwen/Qwen2.5-72B-Instruct \
        --output_dir data/m_arenahard/outputs/Qwen2.5-72B-Instruct \
        --languages en it es de fr \
        --tp_size 8

    # With LoRA (loaded at runtime via vLLM LoRARequest):
    python src/inference_marenahard.py \
        --model_path Qwen/Qwen2.5-72B-Instruct \
        --lora_path /path/to/adapter \
        --output_dir data/m_arenahard/outputs/my-finetuned-model \
        --languages en it es de fr \
        --tp_size 8

    # With LoRA pre-merged into the base model (avoids vLLM LoRA bugs):
    python src/inference_marenahard.py \
        --model_path Qwen/Qwen2.5-72B-Instruct \
        --lora_path /path/to/adapter \
        --merge_lora \
        --output_dir data/m_arenahard/outputs/my-finetuned-model \
        --languages en it es de fr \
        --tp_size 8
"""

import json
import argparse
import os
import shutil
import sys
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ============================================================================
# CONSTANTS
# ============================================================================

AVAILABLE_LANGUAGES = [
    "ar", "cs", "da", "de", "el", "en", "es", "fa", "fr", "he", "hi",
    "id", "it", "ja", "ko", "nl", "pl", "pt", "ro", "ru", "tr",
    "uk", "vi", "zh", "gl", "cy", "mt", "ga"
]


# ============================================================================
# UTILITIES
# ============================================================================

def merge_lora_adapter(adapter_path: str) -> str:
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM
    from huggingface_hub import snapshot_download

    print(f"Merging LoRA adapter: {adapter_path}")

    config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = config.base_model_name_or_path
    print(f"Base model: {base_model_name}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="cpu"
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    merged_path = Path(adapter_path) / "merged"
    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_path)

    base_model_path = Path(base_model_name)
    if not base_model_path.exists():
        base_model_path = Path(snapshot_download(base_model_name))

    tokenizer_patterns = [
        "tokenizer*",
        "special_tokens_map*",
        "added_tokens*",
        "sentencepiece*",
        "vocab*",
        "merges*",
        "chat_template*",
        "generation_config.json",
    ]

    for pattern in tokenizer_patterns:
        for f in base_model_path.glob(pattern):
            shutil.copy2(f, merged_path / f.name)
            print(f"Copied {f.name}")

    print(f"Merged model saved to: {merged_path}")
    return str(merged_path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM inference on m-ArenaHard-v2.0"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to the base model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to a local LoRA adapter directory (optional)"
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge the LoRA adapter into the base model before loading vLLM (avoids vLLM LoRA bugs). Requires --lora_path."
    )
    parser.add_argument(
        "--generator_name",
        type=str,
        default=None,
        help="Name written into the 'generator' field of outputs. Defaults to LoRA dir name if LoRA is used, else base model name."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write per-language output JSON files"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=AVAILABLE_LANGUAGES,
        choices=AVAILABLE_LANGUAGES,
        help="Which language subsets to evaluate (ISO codes)"
    )
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of prompts per language (for debugging)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.merge_lora and not args.lora_path:
        print("Error: --merge_lora requires --lora_path")
        sys.exit(1)

    # Derive generator name before mutating model_path
    if args.generator_name:
        generator_name = args.generator_name
    elif args.lora_path:
        generator_name = args.lora_path.rstrip("/").split("/")[-1]
    else:
        generator_name = args.model_path.rstrip("/").split("/")[-1]

    # Validate LoRA adapter
    if args.lora_path and not os.path.exists(args.lora_path):
        print(f"Error: LoRA adapter not found at '{args.lora_path}'")
        sys.exit(1)

    if args.lora_path:
        adapter_config = os.path.join(args.lora_path, "adapter_config.json")
        adapter_weights = os.path.join(args.lora_path, "adapter_model.safetensors")

        if not os.path.exists(adapter_config):
            print(f"Error: No adapter_config.json found in '{args.lora_path}'")
            sys.exit(1)

        if not os.path.exists(adapter_weights):
            adapter_weights_bin = os.path.join(args.lora_path, "adapter_model.bin")
            if not os.path.exists(adapter_weights_bin):
                print(f"Error: No adapter weights found in '{args.lora_path}' (checked .safetensors and .bin)")
                sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Identify which languages need generation
    languages_to_run = []
    for lang in args.languages:
        out_path = os.path.join(args.output_dir, f"{lang}.json")
        if os.path.exists(out_path):
            print(f"Output file '{out_path}' already exists. Skipping {lang}.")
        else:
            languages_to_run.append(lang)

    if not languages_to_run:
        print("All languages already generated. Nothing to do.")
        return

    # Optionally merge LoRA adapter
    effective_model_path = args.model_path
    use_runtime_lora = bool(args.lora_path) and not args.merge_lora

    if args.merge_lora:
        effective_model_path = merge_lora_adapter(args.lora_path)
        print(f"Loading merged model from: {effective_model_path}")
    else:
        print(f"Loading base model: {args.model_path}")
        if use_runtime_lora:
            print(f"LoRA adapter (runtime): {args.lora_path}")

    # Initialize vLLM
    llm = LLM(
        model=effective_model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_num_seqs=4096,
        enable_prefix_caching=True,
        enable_lora=use_runtime_lora,
    )
    tokenizer = llm.get_tokenizer()

    # Configure sampling parameters
    if "aya" in effective_model_path.lower():
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=args.max_tokens,
        )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            repetition_penalty=1.0,
            max_tokens=args.max_tokens,
        )

    lora_req = (
        LoRARequest("lora_adapter", 1, args.lora_path) if use_runtime_lora else None
    )

    # Log configuration
    print(f"\n  Generator name:     {generator_name}")
    print(f"  Runtime LoRA active: {use_runtime_lora}")
    print(f"  LoRA pre-merged:     {args.merge_lora}")

    if args.lora_path:
        with open(os.path.join(args.lora_path, "adapter_config.json")) as f:
            lora_config = json.load(f)
        print(f"  LoRA rank:           {lora_config.get('r', 'unknown')}")
        print(f"  LoRA alpha:          {lora_config.get('lora_alpha', 'unknown')}")
        print(f"  Target modules:      {lora_config.get('target_modules', 'unknown')}")
        print(f"  Base model in adapter_config: {lora_config.get('base_model_name_or_path', 'unknown')}")

    # Generate per language
    for lang in languages_to_run:
        print(f"\n{'='*60}")
        print(f"  Language: {lang}")
        print(f"{'='*60}")

        # Load dataset
        print(f"  Loading m-ArenaHard-v2.0 subset '{lang}'...")
        ds = load_dataset("CohereLabs/m-ArenaHard-v2.1", lang, split="test")
        split_data = list(ds)

        if args.limit is not None:
            split_data = split_data[: args.limit]
            print(f"  ⚠️  Limiting to first {args.limit} prompts.")

        print(f"  Prompts: {len(split_data)}")

        # Format prompts
        formatted_prompts = []
        for entry in split_data:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": entry["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(prompt)

        # Generate
        print(f"  Starting generation...")
        outputs = llm.generate(
            formatted_prompts,
            sampling_params,
            lora_request=lora_req,
        )

        # Build output records
        records = []
        for entry, request_output in zip(split_data, outputs):
            records.append({
                "question_id": entry["question_id"],
                "prompt": entry["prompt"],
                "output": request_output.outputs[0].text,
                "generator": generator_name,
                "category": entry.get("category", ""),
                "subcategory": entry.get("subcategory", ""),
            })

        # Save
        out_path = os.path.join(args.output_dir, f"{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"  ✅ Saved {len(records)} outputs → {out_path}")

    print(f"\n✅ All done! Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()