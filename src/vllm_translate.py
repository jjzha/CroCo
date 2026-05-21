import argparse
import json
import os
import random
import copy
from vllm import LLM, SamplingParams


# ============================================================================
# LANGUAGE MAPPING
# ============================================================================

LANG_TO_ISO = {
    "English": "en",
    "Danish": "da",
    "Dutch": "nl",
    "German": "de",
    "Italian": "it",
    "Spanish": "es",
    "French": "fr",
    "Portuguese (Portugal)": "pt-PT",
    "Galician": "gl",
    "Welsh": "cy",
    "Maltese": "mt",
    "Irish": "ga"
}


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Translate conversational datasets.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Unbabel/Tower-Plus-72B",
        help="Model ID."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to your English JSON or JSONL file."
    )
    parser.add_argument(
        "--target_languages",
        type=str,
        nargs="+",
        default=["Portuguese (Portugal)", "Spanish", "French"],
        help="Languages to translate into."
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="GPUs to use."
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="Data parallel size."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of random conversations to sample."
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=8192,
        help="Max tokens allowed per individual message."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="translated",
        help="Directory to save translations."
    )
    return parser.parse_args()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_file(data, filepath, is_jsonl):
    with open(filepath, "w", encoding="utf-8") as f:
        if is_jsonl:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(data, f, indent=4, ensure_ascii=False)


def build_prompt(text, target_lang, model_id, tokenizer):
    model_lower = model_id.lower()

    if "translategemma" in model_lower:
        source_iso = "en"
        target_iso = LANG_TO_ISO.get(target_lang, "en")

        raw_content = f"<<<source>>>{source_iso}<<<target>>>{target_iso}<<<text>>>{text}"

        messages = [{"role": "user", "content": raw_content}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    else:
        return f"Translate the following English source text to {target_lang}:\nEnglish: {text}\n{target_lang}: "


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    is_jsonl = args.input_file.lower().endswith(".jsonl")

    # Load English dataset
    print(f"📖 Loading {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        if is_jsonl:
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)

    # Random sampling
    if args.num_samples is not None and args.num_samples > 0:
        if args.num_samples < len(data):
            print(f"🎲 Randomly sampling {args.num_samples} conversations from {len(data)} total...")
            random.seed(42)
            data = random.sample(data, args.num_samples)
        else:
            print(f"⚠️ Requested samples ({args.num_samples}) >= total data ({len(data)}). Using full dataset.")

    os.makedirs("processed_data", exist_ok=True)
    ext = ".jsonl" if is_jsonl else ".json"
    original_output_path = f"processed_data/original_sampled_english{ext}"
    save_file(data, original_output_path, is_jsonl)

    # Initialize vLLM
    print(f"🚀 Initializing vLLM with {args.model_id}...")
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        gpu_memory_utilization=0.95,
        max_num_seqs=1024,
    )
    tokenizer = llm.get_tokenizer()

    # Filter dataset
    longest_lang = max(args.target_languages, key=len)
    print(f"🧹 Filtering dataset: Dropping conversations where ANY message exceeds {args.max_prompt_tokens} tokens...")

    filtered_data = []
    dropped_conversations = 0

    for entry in data:
        is_valid = True
        for msg in entry["messages"]:
            text = msg["content"]
            prompt_str = build_prompt(text, longest_lang, args.model_id, tokenizer)
            token_count = len(tokenizer.encode(prompt_str))

            if token_count > args.max_prompt_tokens:
                is_valid = False
                break

        if is_valid:
            filtered_data.append(entry)
        else:
            dropped_conversations += 1

    print(f"📉 Filtered Data: {len(filtered_data)} conversations kept. {dropped_conversations} whole conversations removed.")

    filtered_output_path = f"processed_data/filtered_english{ext}"
    save_file(filtered_data, filtered_output_path, is_jsonl)

    # Translation loop
    sampling_params = SamplingParams(temperature=0, max_tokens=8192)

    for lang in args.target_languages:
        print(f"\n--- Starting Translation: English -> {lang} ---")

        prompts = []
        mapping = []

        for conv_idx, entry in enumerate(filtered_data):
            for msg_idx, msg in enumerate(entry["messages"]):
                text = msg["content"]
                prompt_str = build_prompt(text, lang, args.model_id, tokenizer)
                prompts.append(prompt_str)
                mapping.append((conv_idx, msg_idx))

        print(f"Batch prepared: {len(prompts)} samples.")

        if not prompts:
            print(f"No prompts for {lang}. Skipping...")
            continue

        outputs = llm.generate(prompts, sampling_params)

        # Reconstruct dataset
        translated_dataset = copy.deepcopy(filtered_data)

        for i, output in enumerate(outputs):
            conv_idx, msg_idx = mapping[i]
            translated_text = output.outputs[0].text.strip()

            translated_dataset[conv_idx]["messages"][msg_idx]["content"] = translated_text
            translated_dataset[conv_idx]["language"] = lang

        # Save translation
        os.makedirs(args.output_dir, exist_ok=True)
        safe_lang = lang.replace(' ', '_').replace('(', '').replace(')', '').lower()
        output_path = os.path.join(args.output_dir, f"{safe_lang}{ext}")

        save_file(translated_dataset, output_path, is_jsonl)
        print(f"✅ Saved {lang} translation to {output_path}")


if __name__ == "__main__":
    main()