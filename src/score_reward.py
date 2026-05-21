import json
import argparse
import os
import sys
import torch
from vllm import LLM, PoolingParams


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2048
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run 1 example only"
    )
    args = parser.parse_args()

    # Pre-run checks
    if not args.debug and os.path.exists(args.output_file):
        print(f"Output file '{args.output_file}' already exists. Skipping.")
        return

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found at '{args.dataset_path}'")
        sys.exit(1)

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading Dataset: {args.dataset_path}")
    raw_data = []

    is_jsonl = args.dataset_path.strip().lower().endswith(".jsonl")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        if is_jsonl:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
        else:
            raw_data = json.load(f)

    if args.debug:
        print("\n=== DEBUG MODE: Processing only 1 entry ===")
        raw_data = raw_data[:1]

    # Initialize vLLM
    print(f"Loading Model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        runner="pooling",
        enforce_eager=True,
        max_model_len=args.max_len,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_num_seqs=2048,
        enable_chunked_prefill=False
    )
    tokenizer = llm.get_tokenizer()

    # Prepare batches
    print("Tokenizing and filtering prompts...")
    valid_entries = []
    batch_inputs = []
    skipped_count = 0

    for entry in raw_data:
        if "messages" in entry:
            full_text = tokenizer.apply_chat_template(
                entry["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            print("Warning: Entry missing 'messages' key, skipping.")
            continue

        token_ids = tokenizer.encode(full_text)

        if len(token_ids) <= args.max_len:
            valid_entries.append(entry)
            batch_inputs.append({"prompt_token_ids": token_ids})
        else:
            skipped_count += 1

    print(f"Prepared {len(batch_inputs)} valid entries.")
    print(f"Skipped {skipped_count} entries (> {args.max_len} tokens).")

    if not batch_inputs:
        print("No valid entries found. Exiting.")
        return

    # Run inference
    print(f"Scoring...")
    outputs = llm.encode(
        prompts=batch_inputs,
        pooling_params=PoolingParams(activation=False),
        pooling_task="classify",
    )

    # Save results
    print(f"Saving results to {args.output_file}...")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for entry, output in zip(valid_entries, outputs):
            score_data = output.outputs.data
            final_score = 0.0

            try:
                if hasattr(score_data, "item"):
                    final_score = score_data.item()
                elif isinstance(score_data, (list, tuple)):
                    final_score = score_data[0]
                else:
                    final_score = float(score_data)
            except Exception as e:
                print(f"Warning: Could not extract score from {score_data}: {e}")
                final_score = -999.0

            res = {
                "id": entry.get("id"),
                "source_dataset": entry.get("source_dataset"),
                "domain": entry.get("domain"),
                "messages": entry.get("messages", []),
                "reward_score": final_score,
                "token_length": len(output.prompt_token_ids),
                "model": args.model_path,
            }

            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("Done!")


if __name__ == "__main__":
    main()