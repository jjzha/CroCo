import argparse
import shutil
from pathlib import Path
from datetime import datetime
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from CroCo.src.euroeval import Benchmarker


# ============================================================================
# UTILITIES
# ============================================================================

def merge_lora_adapter(adapter_path):
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
        "merges*"
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

def perform_benchmark(model_path, dataset_list, langs, output_dir, evaluate_test, merge_lora):
    original_model_name = Path(model_path).name

    if merge_lora:
        model_path = merge_lora_adapter(model_path)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model_name = original_model_name

    results_file = Path(output_dir) / model_name / f"results_{langs}_{timestamp}.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in dataset_list.split(",") if d.strip()]

    benchmarker = Benchmarker()
    benchmarker.results_path = Path(results_file)
    benchmarker.results_path.parent.mkdir(parents=True, exist_ok=True)
    benchmarker.force = True

    print(f"Model: {model_path} | Language: {langs}")
    print(f"Evaluating datasets: {datasets}")

    benchmarker.benchmark(
        model=model_path,
        language=langs,
        dataset=datasets,
        evaluate_test_split=True if evaluate_test else False,
        gpu_memory_utilization=0.95,
        num_iterations=3,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True
    )
    parser.add_argument(
        "--dataset_list",
        type=str,
        required=True
    )
    parser.add_argument(
        "--langs",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results"
    )
    parser.add_argument(
        "--evaluate_test",
        action="store_true",
        help="Evaluate on the test split instead of validation"
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA adapter into base model before benchmarking (avoids vLLM LoRA bugs)"
    )
    args = parser.parse_args()

    perform_benchmark(
        model_path=args.model_id,
        dataset_list=args.dataset_list,
        langs=args.langs,
        output_dir=args.output_dir,
        evaluate_test=args.evaluate_test,
        merge_lora=args.merge_lora,
    )