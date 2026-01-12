"""
LoRA fine-tuning for preference extraction with Qwen3 (causal LM).
The model learns to emit a single preference label given a user request or brief dialog.

Modes:
  - train: fits a LoRA adapter on Qwen3 using synthetic or user-provided data
  - test:  loads a saved adapter and generates a preference label for a prompt

Dataset formats:
  * Synthetic (default): four labels {fast_small, high_quality, lowest_cost, guardrail_first}
  * Custom JSONL: each line -> {"text": "...", "preference": "label"}

Minimal usage:
  python ft_qwen3_preference_lora.py --mode train --model-name Qwen/Qwen3-0.6B-Instruct --output-dir ./qwen3_pref_lora
  python ft_qwen3_preference_lora.py --mode test --model-path ./qwen3_pref_lora --prompt "Need low-latency mobile inference"
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from common_lora_utils import (
    clear_gpu_memory,
    log_memory_usage,
    set_gpu_device,
    setup_logging,
)  # noqa: E402

logger = setup_logging()

DEFAULT_LABELS = ["fast_small", "high_quality", "lowest_cost", "guardrail_first"]


def qwen3_target_modules() -> List[str]:
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def build_instruction(text: str, labels: List[str]) -> str:
    label_str = ", ".join(labels)
    return (
        "You are a routing preference classifier. Given a user request or short dialog, "
        "output exactly one preference label from this set: "
        f"{label_str}. Only return the label.\nRequest: {text}\nAnswer:"
    )


def format_messages(text: str, preference: Optional[str], labels: List[str]):
    instruction = build_instruction(text, labels)
    msgs = [{"role": "user", "content": instruction}]
    if preference is not None:
        msgs.append({"role": "assistant", "content": preference})
    return msgs


@dataclass
class EncodedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    user_length: int
    preference: str


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer,
        labels: List[str],
        max_length: int = 512,
    ):
        self.examples: List[EncodedExample] = []
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        for sample in samples:
            enc = self._encode_sample(sample["text"], sample["preference"])
            self.examples.append(enc)

    def _encode_sample(self, text: str, preference: str) -> EncodedExample:
        messages = format_messages(text, preference, self.labels)
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        full = self.tokenizer(
            formatted, truncation=True, max_length=self.max_length, padding="max_length"
        )

        user_only = format_messages(text, None, self.labels)
        user_text = self.tokenizer.apply_chat_template(
            user_only, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        user_ids = self.tokenizer(user_text, add_special_tokens=False)["input_ids"]
        user_len = len(user_ids)

        labels = full["input_ids"].copy()
        labels[:user_len] = [-100] * min(user_len, len(labels))
        labels = [tok if tok != self.tokenizer.pad_token_id else -100 for tok in labels]

        return EncodedExample(
            input_ids=full["input_ids"],
            attention_mask=full["attention_mask"],
            labels=labels,
            user_length=user_len,
            preference=preference,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(ex.attention_mask, dtype=torch.long),
            "labels": torch.tensor(ex.labels, dtype=torch.long),
            "user_length": ex.user_length,
            "preference": ex.preference,
        }


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "text" not in obj or "preference" not in obj:
                raise ValueError("Each JSONL line must contain 'text' and 'preference'")
            rows.append({"text": obj["text"], "preference": obj["preference"]})
    return rows


def synthetic_dataset() -> List[Dict[str, str]]:
    samples = []
    pref_examples = {
        "fast_small": [
            "Deploy on mobile with strict latency budget",
            "Need sub-second responses on CPU only",
            "Edge device with 4GB RAM, prioritize speed",
            "Low-latency autocomplete for chat UI",
        ],
        "high_quality": [
            "Long-form reasoning with best quality",
            "Research assistant for complex Q&A",
            "Detailed policy analysis",
            "High accuracy summarization for legal docs",
        ],
        "lowest_cost": [
            "Keep tokens cheap for bulk processing",
            "Cost-sensitive batch email triage",
            "Process millions of logs nightly on budget",
            "Optimize for minimal inference cost",
        ],
        "guardrail_first": [
            "Must enforce safety and jailbreak filters",
            "PII-heavy data with strict compliance",
            "Route through strongest safety model",
            "Safety-critical responses for healthcare",
        ],
    }
    for label, texts in pref_examples.items():
        for t in texts:
            samples.append({"text": t, "preference": label})
    random.shuffle(samples)
    return samples


def split_dataset(samples: List[Dict[str, str]], train_ratio: float = 0.8):
    split = int(len(samples) * train_ratio)
    return samples[:split], samples[split:]


def load_datasets(args, tokenizer, labels: List[str]):
    if args.dataset_path:
        rows = load_jsonl(Path(args.dataset_path))
    else:
        rows = synthetic_dataset()
    train_rows, val_rows = split_dataset(rows)
    train_ds = PreferenceDataset(train_rows, tokenizer, labels, args.max_length)
    val_ds = PreferenceDataset(val_rows, tokenizer, labels, args.max_length)
    return train_ds, val_ds


def create_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    return model, tokenizer


def train(args):
    device, _ = set_gpu_device(auto_select=True)
    labels = args.labels.split(",") if args.labels else DEFAULT_LABELS

    model, tokenizer = create_model_and_tokenizer(args.model_name, device)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=qwen3_target_modules(),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds, val_ds = load_datasets(args, tokenizer, labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    log_memory_usage("start")
    trainer.train()
    log_memory_usage("end")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Saved adapter to {args.output_dir}")
    clear_gpu_memory()


def generate_label(
    model, tokenizer, prompt: str, labels: List[str], max_new_tokens: int = 8
) -> str:
    messages = format_messages(prompt, None, labels)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    gen = output[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return decoded.split()[0] if decoded else ""


def load_peft_model(model_path: str):
    cfg = PeftConfig.from_pretrained(model_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base, model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="LoRA preference fine-tune with Qwen3")
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B-Instruct")
    p.add_argument("--model-path", help="Path to trained adapter for test mode")
    p.add_argument("--output-dir", default="./qwen3_preference_lora")
    p.add_argument("--dataset-path", help="Optional JSONL dataset path")
    p.add_argument("--labels", help="Comma-separated label list")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--prompt", help="Prompt for test mode")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train(args)
    else:
        if not args.model_path:
            raise ValueError("--model-path is required in test mode")
        labels = args.labels.split(",") if args.labels else DEFAULT_LABELS
        model, tokenizer = load_peft_model(args.model_path)
        prompt = args.prompt or "Need low-latency model for edge device"
        label = generate_label(model, tokenizer, prompt, labels)
        print(f"Predicted preference: {label}")


if __name__ == "__main__":
    main()
