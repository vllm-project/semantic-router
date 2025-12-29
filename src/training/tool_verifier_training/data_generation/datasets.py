"""
PyTorch Dataset classes for training jailbreak/prompt injection models.

Supports two stages:
- Stage 1: SentinelDataset (Sequence classification: SAFE vs INJECTION_RISK)
- Stage 2: ToolCallVerifierDataset (Token classification for tool-call verification)
"""

import json
from pathlib import Path
from typing import Optional, Literal

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .base import SentinelSample, ToolCallSample


# =============================================================================
# Label Configuration
# =============================================================================


def load_sentinel_label_config(config_path: Optional[str] = None) -> tuple[dict, dict]:
    """Load Stage 1 (Sentinel) label configuration."""

    if config_path and Path(config_path).exists():
        config = json.loads(Path(config_path).read_text())
        label2id = config["label2id"]
        id2label = {int(k): v for k, v in config["id2label"].items()}
        return label2id, id2label

    # Default binary classification labels
    label2id = {
        "SAFE": 0,
        "INJECTION_RISK": 1,
    }
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def load_verifier_label_config(
    config_path: Optional[str] = None,
) -> tuple[dict, dict, dict]:
    """Load Stage 2 (ToolCallVerifier) label configuration."""

    if config_path and Path(config_path).exists():
        config = json.loads(Path(config_path).read_text())
        label2id = config["label2id"]
        id2label = {int(k): v for k, v in config["id2label"].items()}
        severity = config.get("severity", {})
        return label2id, id2label, severity

    # Binary labels for tool-call verification (simpler, more actionable)
    label2id = {
        "AUTHORIZED": 0,  # Tool call aligns with user intent
        "UNAUTHORIZED": 1,  # Violates policy/intent - block it
    }
    id2label = {v: k for k, v in label2id.items()}
    severity = {
        "AUTHORIZED": 0,
        "UNAUTHORIZED": 4,
    }
    return label2id, id2label, severity


# Default labels
SENTINEL_LABEL2ID, SENTINEL_ID2LABEL = load_sentinel_label_config()
VERIFIER_LABEL2ID, VERIFIER_ID2LABEL, VERIFIER_SEVERITY = load_verifier_label_config()


# =============================================================================
# Stage 1: Sentinel Dataset (Sequence Classification)
# =============================================================================


class SentinelDataset(Dataset):
    """PyTorch Dataset for Stage 1 sequence classification."""

    def __init__(
        self,
        samples: list[SentinelSample],
        tokenizer: AutoTokenizer,
        label2id: dict,
        max_length: int = 512,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample.prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        label_id = self.label2id[sample.label]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


# =============================================================================
# Stage 2: Tool-Call Verifier Dataset (Token Classification)
# =============================================================================


class ToolCallVerifierDataset(Dataset):
    """PyTorch Dataset for Stage 2 token-level tool-call verification."""

    def __init__(
        self,
        samples: list[ToolCallSample],
        tokenizer: AutoTokenizer,
        label2id: dict,
        max_length: int = 2048,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.num_labels = len(label2id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Format: [schema + policy + intent] [SEP] [tool_call_json]
        context = f"{sample.tool_schema}\n\nPolicy: {sample.policy}\n\nUser intent: {sample.user_intent}"
        answer = sample.tool_call_json

        encoding = self.tokenizer(
            context,
            answer,
            truncation="only_first",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        offsets = encoding.pop("offset_mapping")[0]

        # Find answer (tool_call) start token
        context_only = self.tokenizer(
            context, add_special_tokens=True, return_tensors="pt"
        )
        answer_start_token = context_only["input_ids"].shape[1]

        # Handle special token edge case
        if (
            answer_start_token < offsets.size(0)
            and offsets[answer_start_token][0] == offsets[answer_start_token][1]
        ):
            answer_start_token += 1

        seq_length = encoding["input_ids"].shape[1]
        labels = [-100] * seq_length

        # Get answer character offset
        answer_char_offset = (
            offsets[answer_start_token][0].item()
            if answer_start_token < len(offsets)
            else 0
        )

        # Default: tool_call tokens are AUTHORIZED
        for i in range(answer_start_token, seq_length):
            labels[i] = self.label2id["AUTHORIZED"]

        # Apply annotations (mark suspicious/unauthorized spans)
        for ann in sample.labels:
            ann_start = ann["start"]
            ann_end = ann["end"]
            ann_label = ann.get("label", ann.get("label_id"))

            if isinstance(ann_label, str):
                if ann_label not in self.label2id:
                    continue
                label_id = self.label2id[ann_label]
            else:
                label_id = ann_label

            # Find overlapping tokens
            for i in range(answer_start_token, seq_length):
                token_start = offsets[i][0].item() - answer_char_offset
                token_end = offsets[i][1].item() - answer_char_offset

                if token_end > ann_start and token_start < ann_end:
                    labels[i] = label_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =============================================================================
# NLI Intent Alignment Dataset
# =============================================================================

from dataclasses import dataclass


@dataclass
class IntentAlignmentSample:
    """A sample for NLI-based intent alignment verification."""

    premise: str  # tool_schema + policy + user_intent
    hypothesis: str  # "The tool call {tool_call} was requested by the user"
    label: Literal["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
    source_dataset: str

    def to_dict(self) -> dict:
        return {
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "label": self.label,
            "source_dataset": self.source_dataset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IntentAlignmentSample":
        return cls(
            premise=d["premise"],
            hypothesis=d["hypothesis"],
            label=d["label"],
            source_dataset=d.get("source_dataset", "unknown"),
        )


class IntentAlignmentDataset(Dataset):
    """PyTorch Dataset for NLI-based intent alignment."""

    NLI_LABEL2ID = {
        "ENTAILMENT": 0,
        "NEUTRAL": 1,
        "CONTRADICTION": 2,
    }

    def __init__(
        self,
        samples: list[IntentAlignmentSample],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = self.NLI_LABEL2ID

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample.premise,
            sample.hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        label_id = self.label2id[sample.label]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


# =============================================================================
# I/O Utilities
# =============================================================================


def load_sentinel_samples(path: str | Path) -> list[SentinelSample]:
    """Load Stage 1 samples from JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return [SentinelSample.from_dict(d) for d in data]


def save_sentinel_samples(samples: list[SentinelSample], path: str | Path):
    """Save Stage 1 samples to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.to_dict() for s in samples]
    path.write_text(json.dumps(data, indent=2))


def load_tool_call_samples(path: str | Path) -> list[ToolCallSample]:
    """Load Stage 2 samples from JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return [ToolCallSample.from_dict(d) for d in data]


def save_tool_call_samples(samples: list[ToolCallSample], path: str | Path):
    """Save Stage 2 samples to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.to_dict() for s in samples]
    path.write_text(json.dumps(data, indent=2))


def load_intent_samples(path: str | Path) -> list[IntentAlignmentSample]:
    """Load NLI samples from JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return [IntentAlignmentSample.from_dict(d) for d in data]


def save_intent_samples(samples: list[IntentAlignmentSample], path: str | Path):
    """Save NLI samples to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.to_dict() for s in samples]
    path.write_text(json.dumps(data, indent=2))
