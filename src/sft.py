import torch
import warnings
import json
import sys
import os
import glob
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# CUSTOM ARGUMENTS
# ============================================================================

@dataclass
class CustomArguments:
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA or full fine-tuning"}
    )
    filter_invalid_conversations: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, drop dataset rows whose 'messages' field would break the "
                "tokenizer's chat template. Validation is performed by actually "
                "running tokenizer.apply_chat_template on each row, so it is "
                "correct for any model (Cohere, Llama, Qwen, Mistral, etc.)."
            )
        },
    )


# ============================================================================
# UTILITIES
# ============================================================================

def make_chat_template_validator(tokenizer):
    def is_valid(example):
        msgs = example.get("messages", None)
        if not isinstance(msgs, list) or len(msgs) == 0:
            return False
        try:
            tokenizer.apply_chat_template(msgs, tokenize=False)
            return True
        except Exception:
            return False

    return is_valid


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_config, custom_args = (
        parser.parse_args_and_config()
    )

    # Load model
    print(f"Loading Model (LoRA: {custom_args.use_lora})...")

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation="flash_attention_2",
        dtype="auto",
        device_map=None,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

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
    else:
        peft_config = None

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Load dataset
    print(f"Loading dataset: {script_args.dataset_name}...")

    if script_args.dataset_name.endswith(".json"):
        print("Detected .json file. Using manual load...")
        with open(script_args.dataset_name, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in data:
                if isinstance(data[key], list):
                    data = data[key]
                    break
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset("json", data_files=script_args.dataset_name)
        if isinstance(dataset, DatasetDict):
            first_key = list(dataset.keys())[0]
            dataset = dataset[first_key]

    print(f"Dataset length: {len(dataset)}")

    # Filter invalid conversations
    if custom_args.filter_invalid_conversations:
        print("Filtering rows that fail tokenizer.apply_chat_template...")
        is_valid = make_chat_template_validator(tokenizer)
        before = len(dataset)
        dataset = dataset.filter(is_valid, num_proc=1)
        after = len(dataset)
        removed = before - after
        pct = (removed / before * 100) if before > 0 else 0.0
        print(
            f"Filtered dataset: {before} -> {after} "
            f"({removed} bad examples removed, {pct:.2f}%)"
        )
        if after == 0:
            raise RuntimeError(
                "All rows were filtered out. Check the structure of your "
                "'messages' field and the model's chat template."
            )

    # Train/eval split
    split_ds = dataset.train_test_split(test_size=0.01)
    train_dataset = split_ds["train"]
    eval_dataset = split_ds["test"]

    print(f"Final Train size: {len(train_dataset)}")
    print(f"Final Eval size: {len(eval_dataset)}")

    # Setup training
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if custom_args.use_lora:
        model.enable_input_require_grads()

    trainer = SFTTrainer(
        model=model,
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

    trainer.train()

    # Save model and metrics
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"SFT Training finished. Model saved to: {training_args.output_dir}")

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
            grad_norm = []
            mean_token_accuracy = []

            for log in log_history:
                if "loss" in log:
                    steps.append(log.get("step"))
                    loss.append(log.get("loss"))
                    if "grad_norm" in log:
                        grad_norm.append(log["grad_norm"])
                    if "mean_token_accuracy" in log:
                        mean_token_accuracy.append(log["mean_token_accuracy"])

            fig, axes = plt.subplots(3, 1, figsize=(10, 15))

            # Loss plot
            if loss:
                axes[0].plot(steps, loss, label="Train Loss", color="#1f77b4", linewidth=2)
                axes[0].set_title("Training Loss Over Steps", fontsize=14)
                axes[0].set_xlabel("Steps")
                axes[0].set_ylabel("Loss")
                axes[0].grid(True, linestyle="--", alpha=0.7)
                axes[0].legend()

            # Gradient norm plot
            if grad_norm:
                axes[1].plot(steps[:len(grad_norm)], grad_norm, label="Gradient Norm", color="#ff7f0e", linewidth=2)
                axes[1].set_title("Gradient Norm Over Steps", fontsize=14)
                axes[1].set_xlabel("Steps")
                axes[1].set_ylabel("Grad Norm")
                axes[1].grid(True, linestyle="--", alpha=0.7)
                axes[1].legend()

            # Mean token accuracy plot
            if mean_token_accuracy:
                axes[2].plot(steps[:len(mean_token_accuracy)], mean_token_accuracy, label="Mean Token Accuracy", color="#2ca02c", linewidth=2)
                axes[2].set_title("Mean Token Accuracy Over Steps", fontsize=14)
                axes[2].set_xlabel("Steps")
                axes[2].set_ylabel("Accuracy")
                axes[2].grid(True, linestyle="--", alpha=0.7)
                axes[2].legend()
            else:
                axes[2].text(
                    0.5,
                    0.5,
                    "Mean Token Accuracy Not Logged",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axes[2].transAxes,
                    fontsize=12,
                    color="gray"
                )
                axes[2].set_axis_off()

            plt.tight_layout()

            plot_path = os.path.join(training_args.output_dir, "training_metrics.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Training metrics plot saved to: {plot_path}")
        else:
            print(f"Warning: Could not find trainer_state.json at {state_path}. Skipping plotting.")