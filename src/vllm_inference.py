import json
import argparse
import os
import sys
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ============================================================================
# UTILITIES
# ============================================================================

def is_mistral_model(model_path: str, tokenizer=None) -> bool:
    name_lower = model_path.lower()
    if "mistral" in name_lower or "mixtral" in name_lower:
        return True
    if tokenizer is not None and "Mistral" in type(tokenizer).__name__:
        return True
    return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the base model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the local LoRA adapter directory (optional)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--language",
        type=str
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        required=True
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of input prompts to process."
    )

    args = parser.parse_args()

    # Pre-run checks
    if os.path.exists(args.output_path):
        print(f"Output file '{args.output_path}' already exists. Skipping inference.")
        return

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found at '{args.dataset_path}'")
        sys.exit(1)

    if args.lora_path and not os.path.exists(args.lora_path):
        print(f"Error: LoRA adapter not found at '{args.lora_path}'")
        sys.exit(1)

    # Initialize vLLM
    print(f"Loading base model: {args.model_path}")

    use_lora = bool(args.lora_path)
    if use_lora:
        print(f"LoRA adapter detected: {args.lora_path}")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_num_seqs=1024,
        enable_prefix_caching=True,
        enable_lora=use_lora,
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset_path}")
    raw_data = []

    file_ext = os.path.splitext(args.dataset_path)[1].lower()

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        if file_ext == ".jsonl":
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
        else:
            raw_data = json.load(f)

    print(f"Total records in file: {len(raw_data)}")

    if args.limit is not None:
        raw_data = raw_data[: args.limit]
        print(f"⚠️ Limiting execution to first {args.limit} prompts.")

    # Detect model type and build prompts
    tokenizer = llm.get_tokenizer()
    use_chat_api = is_mistral_model(args.model_path, tokenizer)

    if use_chat_api:
        print("Detected Mistral-style tokenizer. Using llm.chat() API.")
    else:
        print("Using llm.generate() with manual chat template.")

    formatted_prompts = []
    conversations = []
    final_user_messages = []

    for entry in raw_data:
        user_msgs = [m for m in entry["messages"] if m["role"] == "user"]
        if not user_msgs:
            print("Warning: Skipping entry with no user message.")
            continue

        user_msg = user_msgs[0]["content"]

        if args.language:
            user_msg += f" Please respond in {args.language} and nothing else."

        final_user_messages.append(user_msg)

        if use_chat_api:
            conversations.append([{"role": "user", "content": user_msg}])
        else:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(prompt)

    # Define generation parameters
    sampling_params = SamplingParams(
        n=args.samples,
        max_tokens=4096,
    )

    # Generate completions
    lora_req = LoRARequest("sql_adapter", 1, args.lora_path) if use_lora else None

    if use_chat_api:
        print(f"Starting generation for {len(conversations)} prompts via llm.chat() (Mistral)...")
        outputs = llm.chat(
            conversations,
            sampling_params,
            lora_request=lora_req,
        )
    else:
        print(f"Starting generation for {len(formatted_prompts)} prompts via llm.generate()...")
        outputs = llm.generate(
            formatted_prompts,
            sampling_params,
            lora_request=lora_req,
        )

    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving results to {args.output_path}...")

    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, request_output in enumerate(outputs):
            updated_user_content = final_user_messages[i]
            for completion in request_output.outputs:
                res = {
                    "messages": [
                        {"role": "user", "content": updated_user_content},
                        {"role": "assistant", "content": completion.text},
                    ]
                }
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"✅ Done!")


if __name__ == "__main__":
    main()