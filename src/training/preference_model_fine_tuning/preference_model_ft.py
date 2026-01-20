"""Dataset loader utilities for Qwen3 preference fine-tuning on ShareGPT."""

# TODO:
# 1. Try reranking instead of classification (this is more robust and gives confidence score)
# 2. Try DPO for better alignment

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from dataset_pipeline_sharegpt import ShareGPTConversation, Turn, get_sample_id_hash


CATCH_ALL_LABEL = "general_inquiry"

DEFAULT_SYSTEM_PROMPT = (
    "You are a routing controller that reads a conversation and outputs "
    f"the best preference label for downstream model routing. If none of the labels apply, respond with '{CATCH_ALL_LABEL}'."
)


@dataclass
class PreferenceTrainingExample:
    """Container that pairs a conversation with its canonical label."""

    conversation: ShareGPTConversation
    label: str


def _load_label_mapping(path: Path) -> Dict[str, str]:
    """Load sample_id -> label mapping from JSON or JSONL formats."""

    if not path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {path}")

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
        if isinstance(payload, list):
            mapping: Dict[str, str] = {}
            for item in payload:
                sample_id = str(item.get("sample_id"))
                label = item.get("label")
                if sample_id and label:
                    mapping[sample_id] = str(label)
            return mapping
        raise ValueError("Unsupported JSON label mapping format.")

    mapping: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sample_id = str(item.get("sample_id", "").strip())
            label = str(item.get("label", "").strip())
            if sample_id and label:
                mapping[sample_id] = label
    if not mapping:
        raise ValueError(f"No label mappings found in {path}")
    return mapping


def load_shareGPT_conversation(
    raw_item: Dict[str, object],
) -> Optional[ShareGPTConversation]:
    """Convert a raw ShareGPT JSON record into a ShareGPTConversation."""
    convo = raw_item.get("conversations") or []
    sample_id = str(raw_item.get("id", "").strip())
    if not sample_id:
        return None
    normalized_messages: List[Turn] = []
    for turn in convo:
        speaker_raw = turn.get("from")
        speaker = str(speaker_raw).lower().strip()
        content_raw = turn.get("value")
        content = str(content_raw or "").strip()
        if not content:
            continue
        role = "assistant" if speaker == "gpt" else "user"
        normalized_messages.append(Turn(role=role, content=content))
    if not normalized_messages:
        logging.debug("Skipping sample %s with no valid messages", sample_id)
        return None
    return ShareGPTConversation(sample_id=sample_id, messages=normalized_messages)


def iter_sharegpt_conversations(dataset_path: Path) -> Iterable[ShareGPTConversation]:
    """Yield normalized ShareGPT conversations from the raw dataset."""

    payload = json.loads(dataset_path.read_text())
    for raw_item in payload:
        conversation = load_shareGPT_conversation(raw_item)
        if conversation is not None:
            yield conversation


def build_training_examples(
    dataset_path: Path,
    label_map_path: Path,
    max_samples: Optional[int] = None,
    start_index: int = 0,
) -> List[PreferenceTrainingExample]:
    """Materialize ShareGPT preference examples ready for tokenization."""

    label_mapping = _load_label_mapping(label_map_path)
    examples: List[PreferenceTrainingExample] = []
    for idx, conversation in enumerate(iter_sharegpt_conversations(dataset_path)):
        if idx < start_index:
            continue
        # find matching label by label hash
        sample_id_hash = get_sample_id_hash(conversation.sample_id)
        # TODO: make this less ugly
        label = label_mapping.get(f"{sample_id_hash}_0")
        if not label:
            continue
        examples.append(
            PreferenceTrainingExample(conversation=conversation, label=label)
        )
        if max_samples and len(examples) >= max_samples:
            break
    if not examples:
        raise ValueError(
            "No overlapping samples found between ShareGPT dataset and label map."
        )
    logging.info("Loaded %s labeled ShareGPT samples", len(examples))
    return examples


def conversation_to_text(conversation: ShareGPTConversation) -> str:
    """Serialize a conversation into alternating User/Assistant lines."""

    lines: List[str] = []
    for turn in conversation.messages:
        speaker = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.content}")
    return "\n".join(lines)


def build_prompt(
    conversation: ShareGPTConversation,
    label_space: Sequence[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Construct the natural-language prompt shown to Qwen3."""

    label_clause = ", ".join(sorted(set(label_space)))
    convo_text = conversation_to_text(conversation)
    return (
        f"{system_prompt}\n\n"
        f"Valid labels: {label_clause}\n\n"
        f"Conversation:\n{convo_text}\n\n"
        "Answer with the single label that best matches the conversation."
    )


def format_preference_messages(
    conversation: ShareGPTConversation,
    label_space: Sequence[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    label: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Create ChatML-style messages for preference labeling."""

    label_clause = ", ".join(sorted(set(label_space)))
    user_content = (
        f"Valid labels: {label_clause}\n\n"
        f"Conversation:\n{conversation_to_text(conversation)}\n\n"
        "Answer with the single label that best matches the conversation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if label is not None:
        messages.append({"role": "assistant", "content": label})

    return messages


def build_chat_aligned_dataset(
    examples: Sequence[PreferenceTrainingExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    pad_to_max_length: bool = False,
) -> Dataset:
    """Tokenize examples so only label tokens contribute to loss."""

    input_ids_list: List[List[int]] = []
    attention_mask_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    label_space = sorted({example.label for example in examples})
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    for example in examples:
        prompt_messages = format_preference_messages(
            conversation=example.conversation,
            label_space=label_space,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            label=None,
        )
        full_messages = prompt_messages + [
            {"role": "assistant", "content": example.label}
        ]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        prompt_encoding = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )
        full_encoding = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else False,
        )

        labels = full_encoding["input_ids"].copy()
        prompt_len = len(prompt_encoding["input_ids"])

        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        for i, token_id in enumerate(full_encoding["input_ids"]):
            if token_id == pad_id:
                labels[i] = -100

        input_ids_list.append(full_encoding["input_ids"])
        attention_mask_list.append(full_encoding["attention_mask"])
        labels_list.append(labels)

    return Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }
    )


def compute_token_accuracy(eval_pred: tuple) -> Dict[str, float]:
    """Compute token accuracy on non-masked positions."""

    predictions = (
        eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
    )
    labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]

    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    if predictions.ndim == 3:
        pred_tokens = np.argmax(predictions, axis=-1)
    elif predictions.ndim == 2:
        pred_tokens = predictions
    else:
        return {"token_accuracy": 0.0}

    mask = labels != -100
    correct_tokens = (pred_tokens == labels) & mask
    token_accuracy = correct_tokens.sum() / mask.sum() if mask.sum() > 0 else 0.0

    return {"token_accuracy": float(token_accuracy)}


class ShareGPTPreferenceDataset(TorchDataset):
    """Torch dataset that tokenizes ShareGPT preference examples for Qwen3."""

    def __init__(
        self,
        dataset_path: Path,
        label_map_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        pad_to_max_length: bool = True,
        max_samples: Optional[int] = None,
        start_index: int = 0,
        min_labels_in_prompt: int = 4,
        max_labels_in_prompt: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self.examples = build_training_examples(
            dataset_path=dataset_path,
            label_map_path=label_map_path,
            max_samples=max_samples,
            start_index=start_index,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.system_prompt = system_prompt
        self.label_space = sorted({example.label for example in self.examples})
        if CATCH_ALL_LABEL not in self.label_space:
            # this should not happen
            self.label_space.append(CATCH_ALL_LABEL)
        self.min_labels_in_prompt = max(1, min_labels_in_prompt)
        self.max_labels_in_prompt = (
            max_labels_in_prompt
            if max_labels_in_prompt is None
            else max_labels_in_prompt
        )
        self._rng = np.random.default_rng(random_seed)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        # shuffle the label space to
        # 1. avoid position bias
        # 2. simulate open-set classification
        available_labels = list(self.label_space)

        # clamp bounds to what is feasible for this dataset instance
        max_candidates = len(available_labels)
        upper_bound = (
            max_candidates
            if self.max_labels_in_prompt is None
            else min(self.max_labels_in_prompt, max_candidates)
        )
        lower_bound = min(self.min_labels_in_prompt, upper_bound)

        base_labels = {example.label, CATCH_ALL_LABEL}
        lower_bound = max(lower_bound, len(base_labels))
        upper_bound = max(upper_bound, len(base_labels))

        sample_size = (
            upper_bound
            if lower_bound == upper_bound
            else int(self._rng.integers(lower_bound, upper_bound + 1))
        )

        # ensure the true label and catch-all are always present
        chosen_labels = set(base_labels)
        remaining_needed = sample_size - len(chosen_labels)
        if remaining_needed > 0:
            negative_pool = [
                label for label in available_labels if label not in chosen_labels
            ]
            if negative_pool:
                sampled = self._rng.choice(
                    negative_pool, size=remaining_needed, replace=False
                )
                chosen_labels.update(sampled.tolist())

        randomized_label_space = self._rng.permutation(list(chosen_labels)).tolist()

        prompt = build_prompt(
            conversation=example.conversation,
            label_space=randomized_label_space,
            system_prompt=self.system_prompt,
        )
        tokenized = self.tokenizer(
            prompt,
            text_target=example.label,
            max_length=self.max_length,
            truncation=True,
            padding="max_length" if self.pad_to_max_length else False,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in tokenized.items()}


def build_dataloader(
    dataset: ShareGPTPreferenceDataset,
    batch_size: int,
    shuffle: bool = True,
    collate_fn: Optional[DataCollatorForLanguageModeling] = None,
) -> DataLoader:
    """Create a DataLoader that mirrors Hugging Face Trainer batching behavior."""

    if collate_fn is None:
        collate_fn = DataCollatorForLanguageModeling(
            tokenizer=dataset.tokenizer,
            mlm=False,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a preference classifier on ShareGPT using Hugging Face Trainer."
    )
    parser.add_argument(
        "--sharegpt-path",
        type=Path,
        default=Path("ShareGPT_V3_unfiltered_cleaned_split.json"),
        help="Path to the raw ShareGPT JSON file.",
    )
    parser.add_argument(
        "--label-map-path",
        type=Path,
        default=Path("verified_sharegpt_policy_labels_2.jsonl"),
        help="Path to sample_id -> label mapping (JSON or JSONL).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model identifier (causal LM).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="preference_model_qwen3",
        help="Where to store checkpoints and tokenizer.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit to this many matched samples (None = full dataset).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many samples from the start of the dataset.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Token truncation length for prompts + labels.",
    )
    parser.add_argument(
        "--pad-to-max-length",
        action="store_true",
        help="Pad each example to max_length (else dynamic padding).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for larger effective batch size.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to reserve for validation.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio for scheduler.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=1,
        help="How many checkpoints to keep.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training if supported.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 training if supported.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits and initialization.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    examples = build_training_examples(
        dataset_path=args.sharegpt_path,
        label_map_path=args.label_map_path,
        max_samples=args.max_samples,
        start_index=args.start_index,
    )
    logger.info("Loaded %s labeled examples", len(examples))

    hf_dataset = build_chat_aligned_dataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=args.pad_to_max_length,
    )

    if args.eval_ratio and args.eval_ratio > 0:
        split = hf_dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = hf_dataset
        eval_dataset = None

    model_dtype = (
        torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        report_to=[],
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_token_accuracy if eval_dataset is not None else None,
    )

    logger.info(
        "Starting training with %s train / %s eval examples",
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model and tokenizer saved to %s", args.output_dir)


def main() -> None:  # pragma: no cover - CLI helper
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
