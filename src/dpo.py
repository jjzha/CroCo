from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import sys
import os
import glob
import json
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
from peft import LoraConfig
from trl import (
    ModelConfig,
    ScriptArguments,
    DPOConfig,
    DPOTrainer,
    TrlParser,
)


# ============================================================================
# CUSTOM ARGUMENTS
# ============================================================================

@dataclass
class CustomArguments:
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA or full fine-tuning"}
    )


# ============================================================================
# UTILITIES
# ============================================================================

def is_valid_chat_format(convo):
    if not isinstance(convo, list) or len(convo) == 0:
        return False
    return all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in convo)


def prepare_dpo_format(example):
    chosen = example["chosen"]
    rejected = example["rejected"]

    if isinstance(chosen, str):
        chosen = [{"role": "assistant", "content": chosen}]
    if isinstance(rejected, str):
        rejected = [{"role": "assistant", "content": rejected}]

    if "prompt" in example and example["prompt"]:
        prompt = example["prompt"]
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
    else:
        if len(chosen) > 1:
            prompt = chosen[:-1]
            chosen = [chosen[-1]]

            if len(rejected) > 1:
                rejected = [rejected[-1]]
        else:
            prompt = [{"role": "user", "content": ""}]

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_config, custom_args = (
        parser.parse_args_and_config()
    )

    # Load model
    print(f"Loading Policy Model (LoRA: {custom_args.use_lora})...")

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation="flash_attention_2",
        dtype="auto",
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    model.main_input_name = "input_ids"

    # Configure LoRA
    if custom_args.use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        ref_model = None
    else:
        peft_config = None
        ref_model = None

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Load dataset
    print(f"Loading dataset: {script_args.dataset_name}")
    dataset = load_dataset("json", data_files=script_args.dataset_name)

    if isinstance(dataset, DatasetDict):
        train_ds = (
            dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]
        )
    else:
        train_ds = dataset

    # Validate required columns
    if "chosen" not in train_ds.column_names or "rejected" not in train_ds.column_names:
        print(f"ERROR: Dataset is missing required DPO keys. Found: {train_ds.column_names}")
        print("Dataset must at least contain 'chosen' and 'rejected' columns.")
        sys.exit(1)

    # Prepare DPO format
    print("Enforcing chat template format and extracting prompts if missing...")
    train_ds = train_ds.map(prepare_dpo_format, desc="Formatting and extracting DPO templates")

    # Keep only required columns
    print("Preprocessing dataset: Keeping only required DPO columns...")
    train_ds = train_ds.select_columns(["prompt", "chosen", "rejected"])

    # Validate data format
    print("Validating data format...")
    sample_chosen = train_ds[0]["chosen"]
    sample_rejected = train_ds[0]["rejected"]
    sample_prompt = train_ds[0]["prompt"]

    if not (is_valid_chat_format(sample_chosen) and is_valid_chat_format(sample_rejected) and is_valid_chat_format(sample_prompt)):
        print("\n" + "="*50)
        print("ERROR: Invalid data format detected even after formatting!")
        print("Expected format: [{'role': '...', 'content': '...'}]")
        print(f"Found 'prompt' format: {type(sample_prompt)} - {str(sample_prompt)[:100]}...")
        print("="*50 + "\n")
        sys.exit(1)
    else:
        print("Data format validation passed! It matches the requested structure.")

    # Train/eval split
    split_ds = train_ds.train_test_split(test_size=0.01)
    train_dataset, eval_dataset = split_ds["train"], split_ds["test"]

    print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

    # Setup training
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if custom_args.use_lora:
        model.enable_input_require_grads()

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if (
        hasattr(trainer.accelerator, "distributed_type")
        and trainer.accelerator.distributed_type == "MULTI_GPU"
    ):
        trainer.accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ] = training_args.gradient_accumulation_steps

    print("Starting DPO Training...")
    trainer.train()

    # Save and plot metrics
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"DPO Training finished. Saved to: {training_args.output_dir}")

        # Extract and plot metrics
        print("Extracting metrics from trainer_state.json...")

        checkpoint_dirs = glob.glob(os.path.join(training_args.output_dir, "checkpoint-*"))
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
            state_path = os.path.join(latest_checkpoint, "trainer_state.json")
        else:
            state_path = os.path.join(training_args.output_dir, "trainer_state.json")

        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            log_history = state_data.get("log_history", [])

            steps = []
            loss = []
            rewards_accuracies = []
            rewards_margins = []

            for log in log_history:
                if "loss" in log:
                    steps.append(log.get("step"))
                    loss.append(log.get("loss"))

                    if "rewards/accuracies" in log:
                        rewards_accuracies.append(log["rewards/accuracies"])
                    if "rewards/margins" in log:
                        rewards_margins.append(log["rewards/margins"])

            fig, axes = plt.subplots(3, 1, figsize=(10, 15))

            # Loss plot
            if loss:
                axes[0].plot(steps, loss, label="Train Loss", color="#1f77b4", linewidth=2)
                axes[0].set_title("Training Loss Over Steps", fontsize=14)
                axes[0].set_xlabel("Steps")
                axes[0].set_ylabel("Loss")
                axes[0].grid(True, linestyle="--", alpha=0.7)
                axes[0].legend()

            # Rewards accuracy plot
            if rewards_accuracies:
                axes[1].plot(steps[:len(rewards_accuracies)], rewards_accuracies, label="Rewards Accuracy", color="#2ca02c", linewidth=2)
                axes[1].set_title("Rewards Accuracy Over Steps", fontsize=14)
                axes[1].set_xlabel("Steps")
                axes[1].set_ylabel("Accuracy")
                axes[1].grid(True, linestyle="--", alpha=0.7)
                axes[1].legend()
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "Rewards/Accuracies Not Logged",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[1].transAxes,
                    fontsize=12,
                    color="gray"
                )
                axes[1].set_axis_off()

            # Rewards margin plot
            if rewards_margins:
                axes[2].plot(steps[:len(rewards_margins)], rewards_margins, label="Rewards Margin", color="#ff7f0e", linewidth=2)
                axes[2].set_title("Rewards Margin Over Steps", fontsize=14)
                axes[2].set_xlabel("Steps")
                axes[2].set_ylabel("Margin")
                axes[2].grid(True, linestyle="--", alpha=0.7)
                axes[2].legend()
            else:
                axes[2].text(
                    0.5,
                    0.5,
                    "Rewards/Margins Not Logged",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[2].transAxes,
                    fontsize=12,
                    color="gray"
                )
                axes[2].set_axis_off()

            plt.tight_layout()
            plot_path = os.path.join(training_args.output_dir, "dpo_training_metrics.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"DPO metrics plot saved to: {plot_path}")
        else:
            print(f"Warning: Could not find trainer_state.json at {state_path}. Skipping plotting.")