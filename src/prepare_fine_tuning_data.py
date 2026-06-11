import sys
import json
import argparse
import os
import hashlib
import random

import numpy as np

# ----------------------------------------------------------------------------
# Tasks produced by this script
# ----------------------------------------------------------------------------
#   in_lang_sft   -> SFT, one file per language. Best (max-reward) completion,
#                    prompt + completion in the same language.
#   in_lang_dpo   -> DPO, one file per language. chosen = max reward,
#                    rejected = closest to target (e.g. mu - 2*sigma) within
#                    that same language. Prompt in that language.
#   all_lang_sft  -> SFT, one file. Single global-best (max-reward)
#                    completion per prompt across all languages. Prompt matches
#                    the winning completion's language.
#   all_lang_dpo  -> DPO, one file. chosen = global max reward across
#                    all languages, rejected = closest to target across the whole
#                    cross-lingual pool. Prompt language set by --prompt_source.
# ----------------------------------------------------------------------------

ALL_TASKS = ["in_lang_sft", "in_lang_dpo", "all_lang_sft", "all_lang_dpo"]


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def load_data(filepath, limit=None):
    """Loads records from either a .json array or a .jsonl file.

    If `limit` is set, at most `limit` records are returned (per file) — handy
    for quick debug runs.
    """
    data = []
    if not os.path.exists(filepath):
        print(f"⚠️ WARNING: File not found and will be skipped: {filepath}")
        return data

    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.endswith(".jsonl"):
            for line in f:
                if limit is not None and len(data) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        else:  # .json
            try:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
                else:
                    data.append(content)
            except json.JSONDecodeError:
                pass

    if limit is not None:
        data = data[:limit]
    return data


def get_user_prompt(data):
    """Extracts the user prompt from the messages array."""
    messages = data.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()

    if messages and isinstance(messages, list) and len(messages) > 0:
        return str(messages[0].get("content", "")).strip()
    return None


def get_assistant_completion(data):
    """Extracts the assistant completion from the messages array."""
    messages = data.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()

    if messages and isinstance(messages, list) and len(messages) > 1:
        return str(messages[-1].get("content", "")).strip()
    return ""


def extract_language_from_filename(filename):
    """Extracts the language identifier from the file name."""
    base = os.path.basename(filename).lower()
    for ext in [".jsonl", ".json"]:
        base = base.replace(ext, "")
    base = base.replace("_scored", "")
    return base.split("_")[-1] if "_" in base else base


# ----------------------------------------------------------------------------
# Selection helpers
# ----------------------------------------------------------------------------
def select_chosen(pool):
    """Highest-reward item in the pool."""
    return max(pool, key=lambda x: float(x["reward_score"]))


def select_rejected(pool, chosen, reject_target):
    """Item whose score is closest to the target band and strictly below chosen."""
    scores = [float(x["reward_score"]) for x in pool]

    if reject_target == "mean_minus_2sigma":
        target = np.mean(scores) - 2 * np.std(scores)
    elif reject_target == "q1":
        target = np.percentile(scores, 25)
    else:  # "min"
        target = np.min(scores)

    chosen_score = float(chosen["reward_score"])
    candidates = sorted(pool, key=lambda x: abs(float(x["reward_score"]) - target))
    for candidate in candidates:
        if candidate is not chosen and chosen_score > float(candidate["reward_score"]):
            return candidate
    return None


def resolve_original(original_dict, line_id, desired_lang=None):
    """Pick the parallel original prompt, preferring a specific language."""
    available = original_dict.get(line_id, [])
    if not available:
        return None
    if desired_lang:
        matches = [o for o in available if o.get("_prefix_language") == desired_lang]
        if matches:
            return random.choice(matches)
    return random.choice(available)


# ----------------------------------------------------------------------------
# Record builders
# ----------------------------------------------------------------------------
def make_sft_record(line_id, item, original_dict):
    """SFT example: prompt + best completion in the item's language."""
    lang = item.get("_file_language", "unknown")
    orig = resolve_original(original_dict, line_id, desired_lang=lang)

    if orig:
        prompt = get_user_prompt(orig)
        meta_id = orig.get("id")
        source = orig.get("source_dataset")
        domain = orig.get("domain")
    else:
        prompt = get_user_prompt(item)
        meta_id, source, domain = None, None, None

    pair_hash = hashlib.md5(f"{line_id}_{lang}".encode("utf-8")).hexdigest()[:8]
    return {
        "id": meta_id if meta_id else f"sft_{line_id}_{lang}_{pair_hash}",
        "source_dataset": source,
        "domain": domain,
        "prompt": prompt,
        "language": lang,
        "completion": get_assistant_completion(item),
        "reward_score": float(item["reward_score"]),
    }


def make_dpo_record(line_id, chosen, rejected, original_dict, scope, prompt_source, group_key, pool_size):
    """DPO example: chosen vs rejected, with prompt language resolved per scope."""
    chosen_lang = chosen.get("_file_language", "unknown")
    rejected_lang = rejected.get("_file_language", "unknown")

    if scope == "in_lang":
        # chosen_lang == rejected_lang == group_key
        desired_lang = chosen_lang
    else:  # all_lang
        if prompt_source == "chosen":
            desired_lang = chosen_lang
        elif prompt_source == "rejected":
            desired_lang = rejected_lang
        else:  # random
            desired_lang = None

    orig = resolve_original(original_dict, line_id, desired_lang=desired_lang)

    if orig:
        prompt = get_user_prompt(orig)
        prompt_lang = orig.get("_prefix_language", "unknown")
        meta_id = orig.get("id")
        source = orig.get("source_dataset")
        domain = orig.get("domain")
    else:
        prompt = get_user_prompt(chosen)
        prompt_lang = chosen_lang if scope == "in_lang" else "unknown"
        meta_id, source, domain = None, None, None

    pair_hash = hashlib.md5(f"{line_id}_{group_key}".encode("utf-8")).hexdigest()[:8]
    return {
        "id": meta_id if meta_id else f"dpo_{line_id}_{group_key}_{pair_hash}",
        "source_dataset": source,
        "domain": domain,
        "prompt": prompt,
        "prompt_language": prompt_lang,
        "chosen": get_assistant_completion(chosen),
        "rejected": get_assistant_completion(rejected),
        "chosen_score": float(chosen["reward_score"]),
        "rejected_score": float(rejected["reward_score"]),
        "chosen_language": chosen_lang,
        "rejected_language": rejected_lang,
        "pool_size": pool_size,
    }


# ----------------------------------------------------------------------------
# Task generation
# ----------------------------------------------------------------------------
def partition_by_language(items):
    groups = {}
    for it in items:
        groups.setdefault(it.get("_file_language", "unknown"), []).append(it)
    return groups


def generate_task(task, grouped_data, original_dict, reject_target, prompt_source):
    """Returns (buffers, skipped). buffers maps an output key -> list of records.

    For in_lang_* the key is a language. For all_lang_* the key is "all".
    """
    buffers = {}
    skipped = 0

    for line_id, items in sorted(grouped_data.items()):
        # ---- SFT ----
        if task == "in_lang_sft":
            for lang, pool in partition_by_language(items).items():
                buffers.setdefault(lang, []).append(
                    make_sft_record(line_id, select_chosen(pool), original_dict)
                )

        elif task == "all_lang_sft":
            # Single global-best completion per prompt (across all languages).
            buffers.setdefault("all", []).append(
                make_sft_record(line_id, select_chosen(items), original_dict)
            )

        # ---- DPO ----
        elif task == "in_lang_dpo":
            for lang, pool in partition_by_language(items).items():
                if len(pool) < 2:
                    skipped += 1
                    continue
                chosen = select_chosen(pool)
                rejected = select_rejected(pool, chosen, reject_target)
                if not rejected:
                    skipped += 1
                    continue
                buffers.setdefault(lang, []).append(
                    make_dpo_record(line_id, chosen, rejected, original_dict,
                                    scope="in_lang", prompt_source=prompt_source,
                                    group_key=lang, pool_size=len(pool))
                )

        elif task == "all_lang_dpo":
            pool = items
            if len(pool) < 2:
                skipped += 1
                continue
            chosen = select_chosen(pool)
            rejected = select_rejected(pool, chosen, reject_target)
            if not rejected:
                skipped += 1
                continue
            buffers.setdefault("all", []).append(
                make_dpo_record(line_id, chosen, rejected, original_dict,
                                scope="all_lang", prompt_source=prompt_source,
                                group_key="all", pool_size=len(pool))
            )

    return buffers, skipped


def output_filename(task, key):
    if task == "in_lang_sft":
        return f"sft_in_lang_max_r_{key}.jsonl"
    if task == "in_lang_dpo":
        return f"dpo_in_lang_{key}.jsonl"
    if task == "all_lang_sft":
        return "sft_all_lang_max_r_.jsonl"
    if task == "all_lang_dpo":
        return "dpo_all_lang.jsonl"
    raise ValueError(f"Unknown task: {task}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate per-prompt SFT (max-reward) and DPO (contrastive) data, "
                    "in-language or across languages."
    )
    parser.add_argument("--input_files", type=str, nargs="+", required=True,
                        help="Scored .json/.jsonl files (one per language).")
    parser.add_argument("--original_files", type=str, nargs="+", required=True,
                        help="Parallel original .json/.jsonl files (one per language).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for the generated files.")
    parser.add_argument("--tasks", type=str, nargs="+", choices=ALL_TASKS, default=ALL_TASKS,
                        help="Which datasets to build (default: all four).")
    parser.add_argument("--reject_target", type=str,
                        choices=["mean_minus_2sigma", "q1", "min"], default="mean_minus_2sigma",
                        help="Target band for the DPO 'rejected' completion.")
    parser.add_argument("--prompt_source", type=str,
                        choices=["chosen", "rejected", "random"], default="chosen",
                        help="all_lang_dpo only: which side's language the prompt is taken from.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the combined all_lang files (default: keep line_id order).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Debug: process at most N conceptual prompts (all files are still "
                             "fully read so prompt matching works).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- PASS 0: Map translated prompts to parallel line_ids ---
    prompt_to_line_id = {}
    original_dict = {}
    total_conceptual_prompts = 0

    print("📂 PASS 0: Mapping translated prompts to parallel line_ids...")
    for orig_file in args.original_files:
        prefix_lang = extract_language_from_filename(orig_file)
        original_data = load_data(orig_file)
        total_conceptual_prompts = max(total_conceptual_prompts, len(original_data))

        for line_id, item in enumerate(original_data):
            prompt = get_user_prompt(item)
            if prompt:
                prompt_to_line_id[prompt] = line_id
                original_dict.setdefault(line_id, []).append({**item, "_prefix_language": prefix_lang})

    print(f"   -> Mapped {len(prompt_to_line_id)} translated prompts "
          f"to {total_conceptual_prompts} conceptual line_ids.\n")

    # --- PASS 1: Group scored generations by line_id ---
    print("🔍 PASS 1: Loading scored generations into line_id buckets...")
    grouped_data = {}
    total_lines = 0

    for in_file in args.input_files:
        file_language = extract_language_from_filename(in_file)
        for data in load_data(in_file):
            total_lines += 1
            prompt = get_user_prompt(data)
            if not prompt or data.get("reward_score") is None:
                continue
            line_id = prompt_to_line_id.get(prompt)
            if line_id is None:
                continue
            data["_file_language"] = file_language
            grouped_data.setdefault(line_id, []).append(data)

    if not grouped_data:
        print("❌ Error: No valid scored data found.")
        sys.exit(1)

    # Debug: keep only the first N conceptual prompts (those with matched, scored
    # generations), so the cap is on *prompts* rather than raw input lines.
    if args.max_samples is not None and len(grouped_data) > args.max_samples:
        kept_ids = sorted(grouped_data.keys())[: args.max_samples]
        grouped_data = {lid: grouped_data[lid] for lid in kept_ids}
        print(f"   -> Debug: capped to {len(grouped_data)} conceptual prompts "
              f"(--max_samples={args.max_samples}).")

    # --- PASS 2: Generate each requested task and write files ---
    print(f"⚙️ PASS 2: Building tasks: {', '.join(args.tasks)}\n")
    summary = {}

    for task in args.tasks:
        buffers, skipped = generate_task(
            task, grouped_data, original_dict, args.reject_target, args.prompt_source
        )

        kept = 0
        for key, records in buffers.items():
            # Optionally shuffle the combined cross-lingual files; otherwise
            # everything stays in line_id order (matching the per-language files).
            if args.shuffle and task.startswith("all_lang"):
                random.shuffle(records)

            filepath = os.path.join(args.output_dir, output_filename(task, key))
            with open(filepath, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += len(records)
            print(f"   -> [{task}] saved {len(records)} records to {os.path.basename(filepath)}")

        summary[task] = (kept, skipped)

    # --- Summary ---
    print("\n" + "=" * 55)
    print(f"Total generations processed: {total_lines}")
    print(f"Conceptual prompts found:    {len(grouped_data)} / {total_conceptual_prompts}")
    for task in args.tasks:
        kept, skipped = summary[task]
        print(f"  {task:<14} kept={kept:<7} skipped={skipped}")
    print("=" * 55)


if __name__ == "__main__":
    main()