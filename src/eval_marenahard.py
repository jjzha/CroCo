"""
m-ArenaHard-v2.0 Judge — offline vLLM.

Reads model + reference outputs, constructs pairwise judge prompts,
generates verdicts in one batch, and produces per-language win rates
including length-controlled win rates.

Usage:
    python src/eval_marenahard.py \
        --judge_model_path meta-llama/Llama-3.1-70B-Instruct \
        --model_outputs_dir data/m_arenahard/outputs/my-model \
        --reference_outputs_dir data/m_arenahard/outputs/my-baseline \
        --results_dir data/m_arenahard/results \
        --languages en it es de fr \
        --tp_size 8
"""

import json
import argparse
import os
import re
import sys
import math
import random
from typing import Optional

import numpy as np
from scipy.optimize import minimize
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

JUDGE_PROMPT_WITH_TIES = """\
I need you to compare two responses to the same instruction. Please analyze \
both responses carefully and determine which one is better overall.
<instruction>{instruction}</instruction>
<output_1>{output_1}</output_1>
<output_2>{output_2}</output_2>
Compare the two outputs above. Consider helpfulness, relevance, accuracy, \
depth, creativity, and level of detail. Keep your analysis to at most 500 \
tokens. After your analysis, output your final verdict by strictly following \
this format: "[[1]]" if output_1 is better, "[[2]]" if output_2 is better, \
"[[T]]" if they are equally good. Your final line must be exactly one of: \
[[1]], [[2]], or [[T]]. Do NOT output any other text after your verdict."""

JUDGE_PROMPT_NO_TIES = """\
I need you to compare two responses to the same instruction. Please analyze \
both responses carefully and determine which one is better overall.
<instruction>{instruction}</instruction>
<output_1>{output_1}</output_1>
<output_2>{output_2}</output_2>
Compare the two outputs above. Consider helpfulness, relevance, accuracy, \
depth, creativity, and level of detail. Keep your analysis to at most 500 \
tokens. After your analysis, output your final verdict by strictly following \
this format: "[[1]]" if output_1 is better, "[[2]]" if output_2 is better. \
Your final line must be exactly one of: [[1]] or [[2]]. Do NOT output any \
other text after your verdict."""

_VERDICT_RE = re.compile(r"\[\[\s*([12mMtT])\s*\]\]")


# ============================================================================
# LENGTH-CONTROLLED WIN RATE
# ============================================================================

def compute_lc_winrate(preferences: list[float],
                       model_lengths: list[int],
                       ref_lengths: list[int]) -> dict:
    y = np.array(preferences)
    m_lens = np.array(model_lengths, dtype=float)
    r_lens = np.array(ref_lengths, dtype=float)

    delta = m_lens - r_lens
    std_delta = np.std(delta)

    if std_delta < 1e-8:
        return {
            "lc_win_rate": round(float(np.mean(y)) * 100, 2),
            "theta": 0.0,
            "phi": 0.0,
            "avg_model_len": round(float(np.mean(m_lens))),
            "avg_ref_len": round(float(np.mean(r_lens))),
        }

    x_len = np.tanh(delta / std_delta)
    reg_lambda = 1.0

    def neg_log_likelihood(params):
        theta, phi = params
        logits = theta + phi * x_len
        logits = np.clip(logits, -30, 30)
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        ll = y * np.log(p) + (1 - y) * np.log(1 - p)
        return -np.sum(ll) + reg_lambda * phi**2

    result = minimize(neg_log_likelihood, x0=[0.0, 0.0], method="L-BFGS-B")
    theta, phi = result.x

    lc_wr = 1.0 / (1.0 + math.exp(-theta))

    return {
        "lc_win_rate": round(lc_wr * 100, 2),
        "theta": round(float(theta), 4),
        "phi": round(float(phi), 4),
        "avg_model_len": round(float(np.mean(m_lens))),
        "avg_ref_len": round(float(np.mean(r_lens))),
    }


# ============================================================================
# UTILITIES
# ============================================================================

def build_prompts_for_language(
    model_outputs: list[dict],
    reference_outputs: list[dict],
    tokenizer,
    rng: random.Random,
    judge_prompt: str = JUDGE_PROMPT_WITH_TIES,
    chat_template_kwargs: Optional[dict] = None,
) -> tuple[list[str], list[bool], list[dict]]:
    assert len(model_outputs) == len(reference_outputs)

    ref_by_id = {r["question_id"]: r for r in reference_outputs}
    template_kwargs = chat_template_kwargs or {}

    formatted_prompts = []
    swap_flags = []
    aligned_refs = []
    template_kwargs_failed = False

    for m_out in model_outputs:
        qid = m_out["question_id"]
        r_out = ref_by_id.get(qid)
        assert r_out is not None, (
            f"question_id '{qid}' not found in reference outputs"
        )
        aligned_refs.append(r_out)

        swap = rng.random() < 0.5
        swap_flags.append(swap)

        if swap:
            user_content = judge_prompt.format(
                instruction=m_out["prompt"],
                output_1=r_out["output"],
                output_2=m_out["output"],
            )
        else:
            user_content = judge_prompt.format(
                instruction=m_out["prompt"],
                output_1=m_out["output"],
                output_2=r_out["output"],
            )

        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
        except TypeError as e:
            if template_kwargs and not template_kwargs_failed:
                print(f"  ⚠ tokenizer rejected chat_template_kwargs ({template_kwargs}): {e}. Falling back without them.")
                template_kwargs_failed = True
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )

        formatted_prompts.append(prompt)

    return formatted_prompts, swap_flags, aligned_refs


def thinking_off_template_kwargs(model_path: str) -> dict:
    m = model_path.lower()
    if "qwen3" in m or "qwq" in m:
        return {"enable_thinking": False}
    return {}


def parse_verdict(text: str) -> Optional[str]:
    if not text:
        return None

    cleaned = text.replace("\\[", "[").replace("\\]", "]")
    matches = _VERDICT_RE.findall(cleaned)

    if not matches:
        return None

    last = matches[-1]

    if last == "1":
        return "1"
    if last == "2":
        return "2"

    return "T"


def compute_results(
    model_outputs: list[dict],
    reference_outputs: list[dict],
    outputs,
    swap_flags: list[bool],
    lang: str,
    tie_handling: str = "half",
) -> dict:
    assert tie_handling in ("win", "half", "loss"), \
        f"Invalid tie_handling: {tie_handling}"

    n = len(model_outputs)
    preferences: list[float] = []
    model_lengths: list[int] = []
    ref_lengths: list[int] = []
    annotations = []
    unparseable = 0
    wins = 0
    losses = 0
    ties = 0

    subcat_data = {}

    def score_tie():
        if tie_handling == "win":
            return 1.0, "win"
        if tie_handling == "half":
            return 0.5, "tie"
        return 0.0, "loss"

    for i in range(n):
        out_i = outputs[i].outputs[0]
        response_text = out_i.text
        finish_reason = getattr(out_i, "finish_reason", None)
        verdict = parse_verdict(response_text)
        m_len = len(model_outputs[i]["output"])
        r_len = len(reference_outputs[i]["output"])
        subcat = model_outputs[i].get("subcategory", "unknown")

        if subcat not in subcat_data:
            subcat_data[subcat] = {
                "preferences": [], "model_lengths": [], "ref_lengths": [],
                "wins": 0, "losses": 0, "ties": 0, "unparseable": 0,
            }
        sd = subcat_data[subcat]

        ann = {
            "question_id": model_outputs[i]["question_id"],
            "prompt": model_outputs[i]["prompt"][:200],
            "verdict": verdict,
            "is_swapped": swap_flags[i],
            "model_len": m_len,
            "ref_len": r_len,
            "subcategory": subcat,
            "finish_reason": finish_reason,
        }

        if verdict is None:
            unparseable += 1
            sd["unparseable"] += 1
            ann["preference"] = None
            ann["raw_response"] = response_text
            annotations.append(ann)
            continue

        if verdict == "T":
            preference, bucket = score_tie()
        elif verdict == "1":
            if swap_flags[i]:
                preference, bucket = 0.0, "loss"
            else:
                preference, bucket = 1.0, "win"
        else:
            if swap_flags[i]:
                preference, bucket = 1.0, "win"
            else:
                preference, bucket = 0.0, "loss"

        if bucket == "win":
            wins += 1
            sd["wins"] += 1
        elif bucket == "loss":
            losses += 1
            sd["losses"] += 1
        else:
            ties += 1
            sd["ties"] += 1

        preferences.append(preference)
        model_lengths.append(m_len)
        ref_lengths.append(r_len)
        sd["preferences"].append(preference)
        sd["model_lengths"].append(m_len)
        sd["ref_lengths"].append(r_len)

        ann["preference"] = preference
        annotations.append(ann)

    n_judged = len(preferences)

    if n_judged == 0:
        win_rate = 0.0
        se_pct = 0.0
        lc = {"lc_win_rate": 0.0, "phi": 0.0, "theta": 0.0,
              "avg_model_len": 0, "avg_ref_len": 0}
    else:
        win_rate = sum(preferences) / n_judged * 100
        if n_judged > 1:
            se = (sum((p - win_rate / 100) ** 2 for p in preferences)
                  / (n_judged - 1)) ** 0.5
            se_pct = se / math.sqrt(n_judged) * 100
        else:
            se_pct = 0.0
        lc = compute_lc_winrate(preferences, model_lengths, ref_lengths)

    subcategory_results = {}
    for subcat, sd in sorted(subcat_data.items()):
        sn = len(sd["preferences"])
        if sn == 0:
            subcategory_results[subcat] = {
                "win_rate": 0.0, "lc_win_rate": 0.0, "standard_error": 0.0,
                "n": 0, "n_judged": 0,
                "wins": 0, "losses": 0, "ties": 0,
                "unparseable": sd["unparseable"],
                "avg_model_len": 0, "avg_ref_len": 0,
            }
            continue

        s_wr = sum(sd["preferences"]) / sn * 100
        if sn > 1:
            s_se = (sum((p - s_wr / 100) ** 2 for p in sd["preferences"])
                    / (sn - 1)) ** 0.5
            s_se_pct = s_se / math.sqrt(sn) * 100
        else:
            s_se_pct = 0.0

        s_lc = compute_lc_winrate(
            sd["preferences"], sd["model_lengths"], sd["ref_lengths"])

        subcategory_results[subcat] = {
            "win_rate": round(s_wr, 2),
            "lc_win_rate": s_lc["lc_win_rate"],
            "standard_error": round(s_se_pct, 2),
            "n": sn + sd["unparseable"],
            "n_judged": sn,
            "wins": sd["wins"],
            "losses": sd["losses"],
            "ties": sd["ties"],
            "unparseable": sd["unparseable"],
            "avg_model_len": s_lc["avg_model_len"],
            "avg_ref_len": s_lc["avg_ref_len"],
        }

    return {
        "language": lang,
        "win_rate": round(win_rate, 2),
        "lc_win_rate": lc["lc_win_rate"],
        "standard_error": round(se_pct, 2),
        "n": n,
        "n_judged": n_judged,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "unparseable": unparseable,
        "tie_handling": tie_handling,
        "avg_model_len": lc["avg_model_len"],
        "avg_ref_len": lc["avg_ref_len"],
        "length_bias_coeff_phi": lc["phi"],
        "quality_coeff_theta": lc["theta"],
        "model_generator": model_outputs[0].get("generator", "unknown"),
        "reference_generator": reference_outputs[0].get("generator", "unknown"),
        "subcategory_results": subcategory_results,
        "annotations": annotations,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Judge m-ArenaHard-v2.0 outputs using offline vLLM"
    )
    parser.add_argument(
        "--judge_model_path",
        type=str,
        required=True,
        help="HF model ID or local path for the judge"
    )
    parser.add_argument(
        "--judge_lora_path",
        type=str,
        default=None,
        help="Path to a local LoRA adapter for the judge (optional)"
    )
    parser.add_argument("--model_outputs_dir", type=str, required=True)
    parser.add_argument("--reference_outputs_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=AVAILABLE_LANGUAGES,
        choices=AVAILABLE_LANGUAGES
    )
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens for the judge's verdict generation"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Max context length for the judge model. Prompts exceeding this will have outputs truncated."
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--allow_ties",
        action="store_true",
        default=False,
        help="Allow the judge to declare ties with [[T]]. If off, judge must pick a winner."
    )
    parser.add_argument(
        "--tie_handling",
        type=str,
        default="half",
        choices=["win", "half", "loss"],
        help="How to score a [[T]] tie verdict in win rates: 'win' = tie counts as a full win (preference=1.0, folded into the wins counter); 'half' = tie counts as half credit (preference=0.5, tracked separately in the ties counter — standard AlpacaEval/Arena convention); 'loss' = tie counts as a full loss (preference=0.0, folded into the losses counter). Only matters when --allow_ties is set, since otherwise the judge cannot emit [[T]]. Default: half."
    )
    parser.add_argument(
        "--thinking",
        type=str,
        default="auto",
        choices=["auto", "off", "on"],
        help="Whether to enable the judge's thinking/reasoning mode (Qwen3-style hybrids only). 'auto' (default) = disable thinking if the model name looks like a hybrid-thinking model (qwen3, qwq), otherwise leave the template alone. 'off' = force enable_thinking=False. 'on' = force enable_thinking=True. Disabling is recommended for judging: the verdict is a single tag and a long reasoning preamble just burns max_tokens and risks hitting the length cap."
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    judge_prompt = JUDGE_PROMPT_WITH_TIES if args.allow_ties else JUDGE_PROMPT_NO_TIES
    print(f"Ties: {'allowed' if args.allow_ties else 'disabled'}  (scoring policy: {args.tie_handling})")

    if not args.allow_ties and args.tie_handling != "half":
        print("  Note: --tie_handling has no effect when --allow_ties is off, since the judge cannot emit [[T]].")

    if args.thinking == "auto":
        chat_template_kwargs = thinking_off_template_kwargs(args.judge_model_path)
    elif args.thinking == "off":
        chat_template_kwargs = {"enable_thinking": False}
    else:
        chat_template_kwargs = {"enable_thinking": True}

    if chat_template_kwargs:
        print(f"Chat template kwargs: {chat_template_kwargs}")
    else:
        print("Chat template kwargs: (none — no thinking toggle detected)")

    use_lora = bool(args.judge_lora_path)
    if args.judge_lora_path and not os.path.exists(args.judge_lora_path):
        print(f"Error: LoRA adapter not found at '{args.judge_lora_path}'")
        sys.exit(1)

    print(f"Loading judge model: {args.judge_model_path}")
    if use_lora:
        print(f"LoRA adapter: {args.judge_lora_path}")

    llm = LLM(
        model=args.judge_model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
        max_num_seqs=512,
        enable_prefix_caching=True,
        enable_lora=use_lora,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=args.max_tokens,
    )

    lora_req = (
        LoRARequest("judge_lora", 1, args.judge_lora_path) if use_lora else None
    )

    summary_rows = []

    for lang in args.languages:
        model_path = os.path.join(args.model_outputs_dir, f"{lang}.json")
        ref_path = os.path.join(args.reference_outputs_dir, f"{lang}.json")

        if not os.path.exists(model_path):
            print(f"⚠ Skipping {lang}: not found at {model_path}")
            continue
        if not os.path.exists(ref_path):
            print(f"⚠ Skipping {lang}: not found at {ref_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Judging: {lang}")
        print(f"{'='*60}")

        with open(model_path, "r", encoding="utf-8") as f:
            model_outputs = json.load(f)
        with open(ref_path, "r", encoding="utf-8") as f:
            reference_outputs = json.load(f)

        formatted_prompts, swap_flags, aligned_refs = build_prompts_for_language(
            model_outputs, reference_outputs, tokenizer, rng,
            judge_prompt=judge_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )

        print(f"  Generating {len(formatted_prompts)} judge verdicts...")
        outputs = llm.generate(
            formatted_prompts,
            sampling_params,
            lora_request=lora_req,
        )

        results = compute_results(
            model_outputs, aligned_refs, outputs, swap_flags, lang,
            tie_handling=args.tie_handling,
        )

        detail_path = os.path.join(args.results_dir, f"{lang}_annotations.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        summary_rows.append({
            "language": lang,
            "win_rate": results["win_rate"],
            "lc_win_rate": results["lc_win_rate"],
            "standard_error": results["standard_error"],
            "n": results["n"],
            "n_judged": results["n_judged"],
            "wins": results["wins"],
            "losses": results["losses"],
            "ties": results["ties"],
            "unparseable": results["unparseable"],
            "avg_model_len": results["avg_model_len"],
            "avg_ref_len": results["avg_ref_len"],
            "length_bias_phi": results["length_bias_coeff_phi"],
        })

        unparse_pct = (
            results["unparseable"] / results["n"] * 100 if results["n"] else 0.0
        )
        print(f"  ✅ {lang}:")
        print(f"       win_rate    = {results['win_rate']:.1f}% ± {results['standard_error']:.1f}%  (n_judged={results['n_judged']}/{results['n']})")
        print(f"       lc_win_rate = {results['lc_win_rate']:.1f}%")
        print(f"       W/L/T      = {results['wins']}/{results['losses']}/{results['ties']}")
        print(f"       avg_len model={results['avg_model_len']}  ref={results['avg_ref_len']}  phi={results['length_bias_coeff_phi']:.3f}")

        unparse_marker = "  ⚠" if unparse_pct >= 5.0 else ""
        print(f"       unparseable = {results['unparseable']} ({unparse_pct:.1f}%){unparse_marker}")

        if results["unparseable"] > 0:
            length_cuts = sum(
                1 for a in results["annotations"]
                if a["verdict"] is None and a.get("finish_reason") == "length"
            )
            other_cuts = results["unparseable"] - length_cuts
            print(f"       └─ finish_reason: length={length_cuts}, other={other_cuts}")
            if length_cuts > 0:
                print(f"          (try raising --max_tokens; judge is being cut off)")

        if results["subcategory_results"]:
            print(f"\n       {'Subcategory':<25} {'WR':>7} {'LC WR':>7} {'W':>4} {'L':>4} {'T':>4} {'N(jud)':>9} {'Bad':>4}")
            print(f"       {'-'*25} {'-'*7} {'-'*7} {'-'*4} {'-'*4} {'-'*4} {'-'*9} {'-'*4}")
            for sc, sr in sorted(results["subcategory_results"].items()):
                njud = sr.get("n_judged", sr["n"])
                print(f"       {sc:<25} {sr['win_rate']:>6.1f}% {sr['lc_win_rate']:>6.1f}% {sr['wins']:>4} {sr['losses']:>4} {sr['ties']:>4} {njud:>4}/{sr['n']:<4} {sr['unparseable']:>4}")

    summary_path = os.path.join(args.results_dir, "summary.json")
    summary = {
        "judge_model": args.judge_model_path,
        "model_outputs_dir": args.model_outputs_dir,
        "reference_outputs_dir": args.reference_outputs_dir,
        "seed": args.seed,
        "allow_ties": args.allow_ties,
        "tie_handling": args.tie_handling,
        "thinking": args.thinking,
        "chat_template_kwargs": chat_template_kwargs,
        "results": summary_rows,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Lang':<6} {'Win Rate':>10} {'LC WR':>8} {'± SE':>8} {'W':>5} {'L':>5} {'T':>5} {'Bad':>5} {'N(jud)':>11} {'M Len':>7} {'R Len':>7} {'φ':>7}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*11} {'-'*7} {'-'*7} {'-'*7}")

    for row in summary_rows:
        njud = row.get("n_judged", row["n"])
        print(f"  {row['language']:<6} {row['win_rate']:>9.1f}% {row['lc_win_rate']:>7.1f}% {row['standard_error']:>7.1f}% {row['wins']:>5} {row['losses']:>5} {row['ties']:>5} {row['unparseable']:>5} {njud:>5}/{row['n']:<5} {row['avg_model_len']:>7} {row['avg_ref_len']:>7} {row['length_bias_phi']:>7.3f}")

    if summary_rows:
        avg_wr = sum(r["win_rate"] for r in summary_rows) / len(summary_rows)
        avg_lc = sum(r["lc_win_rate"] for r in summary_rows) / len(summary_rows)
        print(f"  {'-'*6} {'-'*10} {'-'*12}")
        print(f"  {'AVG':<6} {avg_wr:>9.1f}% {avg_lc:>11.1f}%")

    print(f"\n  Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()