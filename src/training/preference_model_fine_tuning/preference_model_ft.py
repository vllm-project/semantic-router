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
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from dataset_pipeline_sharegpt import ShareGPTConversation, Turn, get_sample_id_hash


CATCH_ALL_LABEL = "general_inquiry"

DEFAULT_SYSTEM_PROMPT = (
    "You are a routing controller that reads a conversation and outputs "
    f"the best preference label for downstream model routing. If none of the labels apply, respond with '{CATCH_ALL_LABEL}'\n"
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
        # sample_id_hash = get_sample_id_hash(conversation.sample_id)
        # TODO: make this less ugly
        # label = label_mapping.get(f"{sample_id_hash}_0")
        label = label_mapping.get(conversation.sample_id)
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
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[int]:
    label_clause = ", ".join(sorted(set(label_space)))
    convo_text = conversation_to_text(conversation)
    # truncate the conversation to make sure the result fits in context window
    system_ids = tokenizer(
        system_prompt,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    instruction_ids = tokenizer(
        "\nAnswer with the single label that best matches the conversation.\n",
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    label_ids = tokenizer(
        f"Valid labels:{label_clause}\n\nConversation:",
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    reserved = (
        len(system_ids) + len(instruction_ids) + len(label_ids) + 5
    )  # buffer for target label and end token
    if reserved >= max_length:
        raise ValueError("Instruction + system + labels exceed max_length")
    remaining = max_length - reserved
    conversation_ids = tokenizer(
        convo_text,
        add_special_tokens=False,
        truncation=True,
        max_length=remaining,
    )["input_ids"]

    return system_ids + label_ids + conversation_ids + instruction_ids


def get_random_label_space(
    available_labels: List[str],
    target_label: str,
    max_labels_in_prompt: int,
    min_labels_in_prompt: int,
    rng: np.random.Generator,
) -> List[str]:
    max_candidates = len(available_labels)
    upper_bound = (
        max_candidates
        if max_labels_in_prompt is None
        else min(max_labels_in_prompt, max_candidates)
    )
    lower_bound = min(min_labels_in_prompt, upper_bound)

    base_labels = {target_label, CATCH_ALL_LABEL}
    lower_bound = max(lower_bound, len(base_labels))
    upper_bound = max(upper_bound, len(base_labels))

    sample_size = (
        upper_bound
        if lower_bound == upper_bound
        else int(rng.integers(lower_bound, upper_bound + 1))
    )

    # ensure the true label and catch-all are always present
    chosen_labels = set(base_labels)
    remaining_needed = sample_size - len(chosen_labels)
    if remaining_needed > 0:
        negative_pool = [
            label for label in available_labels if label not in chosen_labels
        ]
        if negative_pool:
            sampled = rng.choice(negative_pool, size=remaining_needed, replace=False)
            chosen_labels.update(sampled.tolist())

    randomized_label_space = rng.permutation(list(chosen_labels)).tolist()
    return randomized_label_space


def build_chat_aligned_dataset(
    examples: Sequence[PreferenceTrainingExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    rng: np.random.Generator,
    max_labels_in_prompt: int,
    min_labels_in_prompt: int,
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
        random_label_space = get_random_label_space(
            available_labels=label_space,
            target_label=example.label,
            max_labels_in_prompt=max_labels_in_prompt,
            min_labels_in_prompt=min_labels_in_prompt,
            rng=rng,
        )
        prompt_encoding = build_prompt(
            conversation=example.conversation,
            label_space=random_label_space,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        full_encoding: list[int] = (
            prompt_encoding
            + tokenizer(
                f"{example.label}<|im_end|>",
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        labels = full_encoding.copy()
        for i in range(len(prompt_encoding)):
            # ignore prompt tokens for label prediction loss
            labels[i] = -100

        # should other paddings be ignored?
        for i, token_id in enumerate(full_encoding):
            # other padding tokens should also be ignored
            if token_id == pad_id:
                labels[i] = -100

        input_ids_list.append(full_encoding)
        # attention mask to mask the padding in input
        attention_mask = [1 if id != pad_id else 0 for id in full_encoding]
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }
    )


def compute_label_accuracy(eval_pred: tuple) -> Dict[str, float]:
    """Compute label accuracy"""

    predictions = (
        eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
    )
    labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]

    # shape of predictions: (batch_size, seq_length)
    # shape of labels: (batch_size, seq_length)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    if predictions.ndim == 3:
        # if we didn't preprocess logits, take argmax over vocab dimension
        pred_tokens = np.argmax(predictions, axis=-1)
    elif predictions.ndim == 2:
        # if we already have token predictions
        pred_tokens = predictions
    else:
        return {"label_accuracy": 0.0}

    mask = labels != -100
    # A sample is correct only if all unmasked tokens match its labels.
    correct_tokens = np.where(mask, pred_tokens == labels, True)
    correct_labels = np.all(correct_tokens, axis=1)
    label_accuracy = float(np.mean(correct_labels)) if correct_labels.size else 0.0

    return {"label_accuracy": label_accuracy}


class LabelPaddingCollator:
    """Pad variable-length causal LM batches, including labels."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # copy to avoid mutating dataset examples
        features = [dict(feature) for feature in features]

        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_length - len(label)
            if pad_len < 0:
                label = label[:max_length]
                pad_len = 0
            padded_labels.append(label + [-100] * pad_len)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


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
        max_labels_in_prompt: Optional[
            int
        ] = 16,  # don't make it too large because otherwise it's hard to fit in memory
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
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation on the base model without training.",
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
    rng = np.random.default_rng(args.seed)
    hf_dataset = build_chat_aligned_dataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_max_length=args.pad_to_max_length,
        max_labels_in_prompt=16,
        min_labels_in_prompt=4,
        rng=rng,
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

    data_collator = LabelPaddingCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="epoch" if eval_dataset is not None else "no",
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

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=-1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_label_accuracy if eval_dataset is not None else None,
    )

    logger.info(
        "Starting training with %s train / %s eval examples",
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
    )
    # If eval-only is requested, skip training and just run evaluation
    if args.eval_only:
        if eval_dataset is None:
            logger.error(
                "Eval-only requested but no eval split found. Set --eval-ratio > 0."
            )
            return
        metrics = trainer.evaluate()
        logger.info("Eval-only metrics: %s", metrics)
        return

    trainer.train()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        logger.info("Eval metrics: %s", metrics)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model and tokenizer saved to %s", args.output_dir)


def main() -> None:  # pragma: no cover - CLI helper
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
