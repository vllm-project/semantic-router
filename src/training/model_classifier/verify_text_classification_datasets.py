#!/usr/bin/env python3
"""
Verify text-classification datasets with single-judge or multi-judge auditing.

This script is intended for the multilingual mmBERT-32K classifier datasets used in
the MoM collection. It supports the text-classification tasks registered in
`src/training/model_eval/constants.py` plus the modality routing dataset exporter:

- feedback
- jailbreak
- fact-check
- intent
- modality

One-stage example:
    python src/training/model_classifier/verify_text_classification_datasets.py \
      --task feedback intent \
      --judge-model minimax2.5 glm5 qwen3.5-397b \
      --api-url http://localhost:8000/v1/chat/completions \
      --sample 500 \
      --correct \
      --confidence high

Two-stage example:
    python src/training/model_classifier/verify_text_classification_datasets.py \
      --task all-text \
      --stage1-model qwen3.5-397b \
      --judge-model kimi-k25 glm5 qwen3.5-397b minimax-m25 \
      --api-url http://localhost:8000/v1/chat/completions \
      --all \
      --correct \
      --confidence high

If different judges live behind different compatible endpoints, override per-model:

    python src/training/model_classifier/verify_text_classification_datasets.py \
      --task feedback \
      --judge-model minimax2.5 glm5 qwen3.5-397b \
      --judge-endpoint minimax2.5=https://gateway-a/v1/chat/completions \
      --judge-endpoint glm5=https://gateway-b/v1/chat/completions \
      --judge-endpoint qwen3.5-397b=https://gateway-c/v1/chat/completions

For locally exported datasets or non-default HF repos, override per task:

    python src/training/model_classifier/verify_text_classification_datasets.py \
      --task modality \
      --dataset-id-override modality=/abs/path/to/modality-routing-dataset/hf_dataset \
      --judge-model qwen3.5-397b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import requests
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CURRENT_DIR.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

try:
    from model_eval.constants import LANGUAGE_CODES, MODEL_REGISTRY
except ImportError:
    from src.training.model_eval.constants import LANGUAGE_CODES, MODEL_REGISTRY


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


MODALITY_DATASET_ID = os.getenv(
    "MODALITY_DATASET_ID",
    "llm-semantic-router/modality-routing-dataset",
)
TEXT_TASKS = tuple(
    sorted(
        {
            *(
                task
                for task, config in MODEL_REGISTRY.items()
                if config["type"] == "text_classification"
            ),
            "modality",
        }
    )
)
CONFIDENCE_SCORES = {"low": 0, "medium": 1, "high": 2}
DEFAULT_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_API_KEY = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")
INTERNAL_SPLIT_COL = "_verify_source_split"
INTERNAL_INDEX_COL = "_verify_source_index"
MAX_VOTE_OBJECT_DEPTH = 10
HIGH_CONFIDENCE_FRACTION = 0.8
MEDIUM_CONFIDENCE_FRACTION = 0.5
PROGRESS_LOG_INTERVAL_SECONDS = 15
LABEL_FIELD_CANDIDATES = (
    "label",
    "predicted_label",
    "modality",
    "category",
    "classification",
    "prediction",
    "class",
)

TASK_INSTRUCTIONS = {
    "feedback": (
        "You are auditing a multilingual user-feedback classifier dataset. "
        "Classify the user message into the single best feedback label."
    ),
    "jailbreak": (
        "You are auditing a multilingual jailbreak-detection dataset. "
        "Classify the text as safe or unsafe based on whether it contains a jailbreak, "
        "policy evasion, or harmful bypass attempt."
    ),
    "fact-check": (
        "You are auditing a multilingual fact-check routing dataset. "
        "Decide whether the text needs fact checking before answering."
    ),
    "intent": (
        "You are auditing a multilingual academic domain classifier dataset. "
        "Choose the single best subject-domain label for the question."
    ),
    "modality": (
        "You are auditing a modality routing dataset. "
        "Choose whether the prompt should route to text-only generation, image generation, "
        "or a multimodal response with both text and image."
    ),
}

TASK_LABEL_HINTS = {
    "feedback": {
        "SAT": "user is satisfied",
        "NEED_CLARIFICATION": "user wants more explanation or detail",
        "WRONG_ANSWER": "user says the answer is wrong or incorrect",
        "WANT_DIFFERENT": "user wants another option, approach, or answer",
    },
    "jailbreak": {
        "safe": "benign text without jailbreak intent",
        "unsafe": "prompt injection, jailbreak, bypass, or malicious evasion request",
    },
    "fact-check": {
        "NO_FACT_CHECK_NEEDED": "does not require factual verification routing",
        "FACT_CHECK_NEEDED": "contains claims or content that should be fact checked",
    },
    "modality": {
        "AR": "text-only response via an autoregressive LLM",
        "DIFFUSION": "image generation via a diffusion model",
        "BOTH": "best served by both text and image output",
    },
}


@dataclass(frozen=True)
class TaskSpec:
    task: str
    dataset_id: str
    split: str
    text_col: str
    label_col: str
    labels: Sequence[str]
    label_storage: str
    label_name_col: str | None = None


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    api_url: str
    api_key: str | None = None


@dataclass
class JudgeVote:
    model: str
    predicted_label: str
    confidence: str
    reasoning: str
    is_correct: bool | None
    raw_response: dict | None = None
    error: str | None = None


@dataclass
class VerificationResult:
    index: int
    task: str
    source_split: str
    source_index: int
    text: str
    original_label: str
    predicted_label: str
    confidence: str
    is_correct: bool
    suggested_correction: str | None
    vote_count: int
    total_votes: int
    vote_fraction: float
    judge_votes: list[JudgeVote] = field(default_factory=list)
    stage1_vote: JudgeVote | None = None
    stage2_trigger_reasons: list[str] = field(default_factory=list)
    stage2_judge_votes: list[JudgeVote] = field(default_factory=list)
    review_path: str = "single_stage"


@dataclass
class TaskStats:
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    uncertain: int = 0
    errors: int = 0
    label_stats: dict[str, dict[str, dict[str, int] | int]] = field(
        default_factory=dict
    )

    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "uncertain": self.uncertain,
            "errors": self.errors,
            "accuracy": self.accuracy(),
            "label_stats": self.label_stats,
        }


TASK_SPECS: dict[str, TaskSpec] = {
    "feedback": TaskSpec(
        task="feedback",
        dataset_id=MODEL_REGISTRY["feedback"]["hf_dataset"],
        split=MODEL_REGISTRY["feedback"]["split"],
        text_col="text",
        label_col="label",
        label_name_col="label_name",
        labels=tuple(MODEL_REGISTRY["feedback"]["labels"]),
        label_storage="index",
    ),
    "jailbreak": TaskSpec(
        task="jailbreak",
        dataset_id=MODEL_REGISTRY["jailbreak"]["hf_dataset"],
        split=MODEL_REGISTRY["jailbreak"]["split"],
        text_col="text",
        label_col=MODEL_REGISTRY["jailbreak"]["label_col"],
        labels=tuple(MODEL_REGISTRY["jailbreak"]["labels"]),
        label_storage="index",
    ),
    "fact-check": TaskSpec(
        task="fact-check",
        dataset_id=MODEL_REGISTRY["fact-check"]["hf_dataset"],
        split=MODEL_REGISTRY["fact-check"]["split"],
        text_col="text",
        label_col=MODEL_REGISTRY["fact-check"]["label_col"],
        labels=tuple(MODEL_REGISTRY["fact-check"]["labels"]),
        label_storage="index",
    ),
    "intent": TaskSpec(
        task="intent",
        dataset_id=MODEL_REGISTRY["intent"]["hf_dataset"],
        split=MODEL_REGISTRY["intent"]["split"],
        text_col="question",
        label_col="category",
        label_name_col="category",
        labels=tuple(MODEL_REGISTRY["intent"]["labels"]),
        label_storage="name",
    ),
    "modality": TaskSpec(
        task="modality",
        dataset_id=MODALITY_DATASET_ID,
        split="validation",
        text_col="text",
        label_col="label",
        label_name_col="label_name",
        labels=("AR", "DIFFUSION", "BOTH"),
        label_storage="index",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify text-classification datasets using one-stage or two-stage LLM auditing."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        required=True,
        choices=[*TEXT_TASKS, "all-text"],
        help="Dataset task(s) to verify. Use all-text to verify every text-classification task.",
    )
    parser.add_argument(
        "--judge-model",
        nargs="+",
        default=None,
        help="One or more LLM judge model names. In one-stage mode these are the full-sweep judges. In two-stage mode these are the stage2 voting judges.",
    )
    parser.add_argument(
        "--judge-endpoint",
        action="append",
        default=[],
        help="Optional per-model endpoint override in MODEL=URL format.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="Default OpenAI-compatible chat-completions endpoint.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="Optional bearer token for the compatible endpoint.",
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        default=None,
        help="Optional stage1 judge model for fast full-sweep prescreening.",
    )
    parser.add_argument(
        "--stage1-api-url",
        type=str,
        default=None,
        help="Optional endpoint override for the stage1 judge. Defaults to --api-url.",
    )
    parser.add_argument(
        "--stage1-api-key",
        type=str,
        default=None,
        help="Optional bearer token override for the stage1 judge. Defaults to --api-key.",
    )
    parser.add_argument(
        "--stage1-escalate-confidence",
        type=str,
        choices=list(CONFIDENCE_SCORES.keys()),
        default="low",
        help="Escalate stage1 samples whose confidence is at or below this level. Default low keeps stage2 focused on low-confidence or label-mismatch cases.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split override for every selected task. Defaults to each task's registry split.",
    )
    parser.add_argument(
        "--dataset-id-override",
        action="append",
        default=[],
        help="Optional per-task dataset override in TASK=DATASET_OR_PATH format. For local saved datasets, point to the exported hf_dataset/ directory or its parent export directory.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=500,
        help="Number of examples to verify per task. Ignored when --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify the full selected split instead of sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling examples.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel examples to verify.",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=list(LANGUAGE_CODES.keys()),
        default=None,
        help="Optional language filter for multilingual error analysis.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum characters from each example sent to the judge.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for each judge request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Optional max response tokens for each judge request. Set to 0 or a negative value to omit the field entirely, which is safer for reasoning-heavy models.",
    )
    parser.add_argument(
        "--correct",
        action="store_true",
        help="Apply high/medium-confidence voted corrections and save a corrected dataset.",
    )
    parser.add_argument(
        "--confidence",
        type=str,
        choices=list(CONFIDENCE_SCORES.keys()),
        default="high",
        help="Minimum ensemble confidence required for applying corrections.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/training/model_classifier/verified_datasets_vote",
        help="Directory for reports and corrected datasets.",
    )
    return parser.parse_args()


def resolve_tasks(tasks: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for task in tasks:
        if task == "all-text":
            resolved.extend(TEXT_TASKS)
        else:
            resolved.append(task)
    return sorted(set(resolved))


def parse_mapping_args(entries: Iterable[str], name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid {name} entry '{entry}'. Expected KEY=VALUE.")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid {name} entry '{entry}'. Expected KEY=VALUE.")
        parsed[key] = value
    return parsed


def build_judge_configs(
    model_names: Sequence[str] | None,
    default_api_url: str,
    default_api_key: str | None,
    endpoint_entries: Iterable[str],
) -> list[JudgeConfig]:
    endpoint_overrides = parse_mapping_args(endpoint_entries, "--judge-endpoint")
    judges = []
    for model_name in model_names or []:
        judges.append(
            JudgeConfig(
                model=model_name,
                api_url=endpoint_overrides.get(model_name, default_api_url),
                api_key=default_api_key,
            )
        )
    return judges


def build_task_specs(
    tasks: Sequence[str],
    dataset_override_entries: Iterable[str],
) -> dict[str, TaskSpec]:
    dataset_overrides = parse_mapping_args(
        dataset_override_entries,
        "--dataset-id-override",
    )
    unknown_tasks = sorted(set(dataset_overrides) - set(TASK_SPECS))
    if unknown_tasks:
        raise ValueError(
            "Unknown task(s) for --dataset-id-override: " + ", ".join(unknown_tasks)
        )

    specs: dict[str, TaskSpec] = {}
    for task in tasks:
        spec = TASK_SPECS[task]
        override_source = dataset_overrides.get(task)
        if override_source:
            spec = replace(spec, dataset_id=override_source)
        specs[task] = spec
    return specs


def retry_load_dataset(load_fn, max_retries: int = 3, delay: float = 2.0):
    for attempt in range(max_retries):
        try:
            return load_fn()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay * (2**attempt))


def detect_language(text: str, target_lang: str) -> bool:
    if not text:
        return False

    script_checks = {
        "zh": lambda t: any("\u4e00" <= char <= "\u9fff" for char in t),
        "ja": lambda t: any("\u3040" <= char <= "\u30ff" for char in t),
        "ar": lambda t: any("\u0600" <= char <= "\u06ff" for char in t),
        "hi": lambda t: any("\u0900" <= char <= "\u097f" for char in t),
    }
    if target_lang in script_checks:
        return script_checks[target_lang](text)

    return any("a" <= char.lower() <= "z" for char in text)


def get_text(example: dict, spec: TaskSpec) -> str:
    return str(example.get(spec.text_col, "") or "")


def normalize_label_name(example: dict, spec: TaskSpec) -> str | None:
    if spec.label_name_col:
        label_name = example.get(spec.label_name_col)
        if isinstance(label_name, str) and label_name in spec.labels:
            return label_name

    raw_value = example.get(spec.label_col)
    if isinstance(raw_value, str):
        if raw_value in spec.labels:
            return raw_value
        if raw_value.isdigit():
            raw_value = int(raw_value)
    if isinstance(raw_value, (int, float)):
        index = int(raw_value)
        if 0 <= index < len(spec.labels):
            return spec.labels[index]
    return None


def filter_dataset_by_language(
    dataset: Dataset, spec: TaskSpec, language: str
) -> Dataset:
    original_len = len(dataset)
    filtered = dataset.filter(lambda x: detect_language(get_text(x, spec), language))
    logger.info(
        "Language filter (%s) for %s: %s -> %s samples",
        language,
        spec.task,
        original_len,
        len(filtered),
    )
    return filtered


def load_task_dataset(  # noqa: C901,PLR0912,PLR0915
    spec: TaskSpec,
    split_override: str | None,
    language: str | None,
) -> tuple[Dataset, list[str], str]:
    requested_split = split_override or spec.split
    dataset_source = spec.dataset_id
    dataset_path = Path(dataset_source).expanduser()
    nested_disk_path = dataset_path / "hf_dataset"
    disk_dataset_path: Path | None = None

    if nested_disk_path.exists():
        disk_dataset_path = nested_disk_path
    elif dataset_path.exists():
        disk_dataset_path = dataset_path

    if disk_dataset_path is not None:
        logger.info(
            "Loading %s dataset from local disk at %s",
            spec.task,
            disk_dataset_path,
        )
        loaded = load_from_disk(str(disk_dataset_path))

        if isinstance(loaded, DatasetDict):
            dataset_dict = loaded

            if requested_split == "all-splits":
                merged_splits = []
                available_splits: list[str] = []
                for split_name in dataset_dict:
                    split_dataset = prepare_split_dataset(
                        dataset=dataset_dict[split_name],
                        spec=spec,
                        split_name=split_name,
                        language=language,
                    )
                    if len(split_dataset) == 0:
                        continue
                    merged_splits.append(split_dataset)
                    available_splits.append(split_name)

                if not merged_splits:
                    raise ValueError(
                        f"No usable samples found for task {spec.task} across all local splits"
                    )
                return (
                    concatenate_datasets(merged_splits),
                    available_splits,
                    requested_split,
                )

            if requested_split not in dataset_dict:
                raise ValueError(
                    f"Split '{requested_split}' not found for task {spec.task} in {disk_dataset_path}. "
                    f"Available splits: {sorted(dataset_dict.keys())}"
                )

            prepared = prepare_split_dataset(
                dataset=dataset_dict[requested_split],
                spec=spec,
                split_name=requested_split,
                language=language,
            )
            if len(prepared) == 0:
                raise ValueError(f"No usable samples found for task {spec.task}")
            return prepared, [requested_split], requested_split

        if requested_split == "all-splits":
            raise ValueError(
                f"Task {spec.task} local source {disk_dataset_path} is a single Dataset, not a DatasetDict. "
                "Use a concrete split or point to a saved DatasetDict."
            )

        prepared = prepare_split_dataset(
            dataset=loaded,
            spec=spec,
            split_name=requested_split,
            language=language,
        )
        if len(prepared) == 0:
            raise ValueError(f"No usable samples found for task {spec.task}")
        return prepared, [requested_split], requested_split

    if requested_split == "all-splits":
        logger.info("Loading %s all splits from %s", spec.task, spec.dataset_id)
        dataset_dict = retry_load_dataset(
            lambda: load_dataset(spec.dataset_id),
            max_retries=3,
        )

        merged_splits = []
        available_splits: list[str] = []
        for split_name in dataset_dict:
            split_dataset = prepare_split_dataset(
                dataset=dataset_dict[split_name],
                spec=spec,
                split_name=split_name,
                language=language,
            )
            if len(split_dataset) == 0:
                continue
            merged_splits.append(split_dataset)
            available_splits.append(split_name)

        if not merged_splits:
            raise ValueError(
                f"No usable samples found for task {spec.task} across all splits"
            )
        return concatenate_datasets(merged_splits), available_splits, requested_split

    logger.info(
        "Loading %s split '%s' from %s", spec.task, requested_split, spec.dataset_id
    )
    dataset = retry_load_dataset(
        lambda: load_dataset(spec.dataset_id, split=requested_split),
        max_retries=3,
    )
    prepared = prepare_split_dataset(
        dataset=dataset,
        spec=spec,
        split_name=requested_split,
        language=language,
    )
    if len(prepared) == 0:
        raise ValueError(f"No usable samples found for task {spec.task}")
    return prepared, [requested_split], requested_split


def prepare_split_dataset(
    dataset: Dataset,
    spec: TaskSpec,
    split_name: str,
    language: str | None,
) -> Dataset:
    if spec.task == "intent":
        dataset = dataset.filter(lambda x: x.get("category") in spec.labels)

    if language:
        dataset = filter_dataset_by_language(dataset, spec, language)

    dataset = dataset.filter(lambda x: normalize_label_name(x, spec) is not None)
    if len(dataset) == 0:
        return dataset

    if INTERNAL_SPLIT_COL not in dataset.column_names:
        dataset = dataset.add_column(INTERNAL_SPLIT_COL, [split_name] * len(dataset))
    if INTERNAL_INDEX_COL not in dataset.column_names:
        dataset = dataset.add_column(INTERNAL_INDEX_COL, list(range(len(dataset))))
    return dataset


def choose_indices(
    dataset_size: int, sample_size: int | None, use_all: bool, seed: int
) -> list[int]:
    if use_all or sample_size is None or sample_size >= dataset_size:
        return list(range(dataset_size))
    ordered = list(range(dataset_size))
    rng = __import__("random").Random(seed)
    rng.shuffle(ordered)
    return sorted(ordered[:sample_size])


def build_label_block(task: str, labels: Sequence[str]) -> str:
    hints = TASK_LABEL_HINTS.get(task, {})
    lines = []
    for label in labels:
        hint = hints.get(label)
        if hint:
            lines.append(f"- {label}: {hint}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


def create_verification_prompt(
    task: str,
    labels: Sequence[str],
    text: str,
    original_label: str,
    max_chars: int,
) -> str:
    text_snippet = text[:max_chars].replace("```", "'''").strip()
    label_block = build_label_block(task, labels)
    task_instruction = TASK_INSTRUCTIONS[task]
    return f"""\
{task_instruction}

Allowed labels:
{label_block}

Current label: {original_label}

Text:
\"\"\"
{text_snippet}
\"\"\"

Return JSON only:
{{
  "label": "<one allowed label>",
  "correct": true or false,
  "confidence": "high" | "medium" | "low",
  "reasoning": "short reason"
}}"""


def extract_message_content(response_json: dict) -> str:  # noqa: C901,PLR0912
    choices = response_json.get("choices") or []
    if not choices:
        for key in ("output_text", "text", "content"):
            value = response_json.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    message = choices[0].get("message") or {}
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)

    content = message.get("content", "")
    if content is None:
        content = ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text", "input_text"}:
                    chunks.append(str(item.get("text", "")))
                elif "text" in item:
                    chunks.append(str(item["text"]))
                elif "content" in item:
                    chunks.append(str(item["content"]))
            else:
                chunks.append(str(item))
        return "".join(chunks)

    if choices[0].get("text"):
        return str(choices[0]["text"])
    return str(content)


def extract_json_from_response(content: str) -> dict | None:  # noqa: C901,PLR0912
    content = content.strip()
    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    if "```" in content:
        for block in content.split("```")[1::2]:
            candidate = block.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    lowered = content.lower()
    for marker in ("assistantfinal", "assistant_final", "final:", "output:"):
        marker_idx = lowered.find(marker)
        if marker_idx == -1:
            continue
        remaining = content[marker_idx + len(marker) :]
        brace_idx = remaining.find("{")
        if brace_idx == -1:
            continue
        candidate = extract_balanced_json(remaining[brace_idx:])
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    brace_idx = content.find("{")
    if brace_idx != -1:
        candidate = extract_balanced_json(content[brace_idx:])
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    return None


def extract_balanced_json(text: str) -> str | None:
    brace_count = 0
    for index, char in enumerate(text):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[: index + 1]
    return None


def extract_vote_object(  # noqa: C901,PLR0911,PLR0912
    value, depth: int = 0
) -> dict | None:
    if depth > MAX_VOTE_OBJECT_DEPTH or value is None:
        return None

    if isinstance(value, dict):
        if any(key in value for key in LABEL_FIELD_CANDIDATES):
            return value

        priority_keys = (
            "parsed",
            "arguments",
            "function_call",
            "function",
            "tool_calls",
            "message",
            "output",
            "result",
            "response",
            "data",
            "content",
            "text",
        )
        for key in priority_keys:
            if key not in value:
                continue
            parsed = extract_vote_object(value.get(key), depth + 1)
            if parsed is not None:
                return parsed

        for nested in value.values():
            parsed = extract_vote_object(nested, depth + 1)
            if parsed is not None:
                return parsed
        return None

    if isinstance(value, list):
        for item in value:
            parsed = extract_vote_object(item, depth + 1)
            if parsed is not None:
                return parsed
        return None

    if isinstance(value, str):
        parsed = extract_json_from_response(value)
        if parsed is None:
            return None
        nested = extract_vote_object(parsed, depth + 1)
        return nested or parsed

    return None


def summarize_response_shape(response_json: dict) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return f"top_level_keys={sorted(response_json.keys())[:10]}"

    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        content_preview = content[:120]
    elif isinstance(content, list):
        content_preview = f"list[{len(content)}]"
    elif content is None:
        content_preview = "None"
    else:
        content_preview = str(content)[:120]

    return (
        f"choice_keys={sorted(choice.keys())[:10]} "
        f"message_keys={sorted(message.keys())[:10]} "
        f"content_preview={content_preview!r}"
    )


def extract_reasoning_content(response_json: dict) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        return reasoning
    if isinstance(reasoning, list):
        chunks = []
        for item in reasoning:
            if isinstance(item, dict):
                if "text" in item:
                    chunks.append(str(item["text"]))
                elif "content" in item:
                    chunks.append(str(item["content"]))
            else:
                chunks.append(str(item))
        return "".join(chunks)
    return ""


def extract_vote_from_response_json(response_json: dict) -> dict | None:
    direct_vote = extract_vote_object(response_json)
    if direct_vote is not None:
        return direct_vote

    content = extract_message_content(response_json)
    if content:
        return extract_vote_object(content)
    return None


def normalize_label_candidate(predicted_label, labels: Sequence[str]) -> str | None:
    if predicted_label is None:
        return None

    candidate = str(predicted_label).strip()
    if not candidate:
        return None
    if candidate in labels:
        return candidate

    label_map = {label.lower(): label for label in labels}
    return label_map.get(candidate.lower())


def parse_correct_flag(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return None


def serialize_judge_config(judge: JudgeConfig) -> dict[str, object]:
    return {
        "model": judge.model,
        "api_url": judge.api_url,
        "api_key_present": bool(judge.api_key),
    }


def call_judge(
    judge: JudgeConfig,
    prompt: str,
    timeout: int,
    max_retries: int,
    max_tokens: int,
) -> tuple[dict | None, str | None]:
    headers = {"Content-Type": "application/json"}
    if judge.api_key:
        headers["Authorization"] = f"Bearer {judge.api_key}"

    base_payload = {
        "model": judge.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict dataset auditor. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    use_json_mode = True
    for attempt in range(max_retries):
        try:
            payload = dict(base_payload)
            if max_tokens and max_tokens > 0:
                payload["max_tokens"] = max_tokens
            if use_json_mode:
                payload["response_format"] = {"type": "json_object"}

            response = requests.post(
                judge.api_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            if use_json_mode and response.status_code in {400, 404, 422}:
                use_json_mode = False
                continue
            response.raise_for_status()
            response_json = response.json()
            parsed = extract_vote_from_response_json(response_json)
            if parsed is not None:
                return parsed, None

            content = extract_message_content(response_json)
            reasoning_content = extract_reasoning_content(response_json)
            finish_reason = None
            choices = response_json.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason")

            if not content and finish_reason == "length":
                error = (
                    "Model stopped with finish_reason=length before emitting final content. "
                    "Retry without --max-tokens or with a much larger limit. "
                    f"reasoning_preview={reasoning_content[:200]!r}; "
                    f"{summarize_response_shape(response_json)}"
                )
                continue

            error = (
                "Could not parse JSON from response: "
                f"{content[:200] if content else '[empty content]'}; "
                f"finish_reason={finish_reason!r}; "
                f"reasoning_preview={reasoning_content[:200]!r}; "
                f"{summarize_response_shape(response_json)}"
            )
        except requests.RequestException as exc:
            error = f"Request failed: {exc}"
        except Exception as exc:
            error = f"Unexpected error: {exc}"

        if attempt < max_retries - 1:
            time.sleep(2**attempt)

    return None, error


def normalize_vote(
    raw_vote: dict | None,
    error: str | None,
    labels: Sequence[str],
    judge: JudgeConfig,
) -> JudgeVote:
    if raw_vote is None:
        return JudgeVote(
            model=judge.model,
            predicted_label="ERROR",
            confidence="low",
            reasoning=error or "judge call failed",
            is_correct=None,
            error=error,
        )

    predicted_label = normalize_label_candidate(
        raw_vote.get("label")
        or raw_vote.get("predicted_label")
        or raw_vote.get("modality")
        or raw_vote.get("category")
        or raw_vote.get("classification")
        or raw_vote.get("prediction")
        or raw_vote.get("class"),
        labels,
    )
    confidence = str(raw_vote.get("confidence", "medium")).lower()
    reasoning = str(raw_vote.get("reasoning", "")).strip()
    is_correct = parse_correct_flag(raw_vote.get("correct"))

    if predicted_label not in labels:
        return JudgeVote(
            model=judge.model,
            predicted_label="ERROR",
            confidence="low",
            reasoning=f"invalid label: {predicted_label}",
            is_correct=None,
            raw_response=raw_vote,
            error="invalid label",
        )

    if confidence not in CONFIDENCE_SCORES:
        confidence = "medium"

    return JudgeVote(
        model=judge.model,
        predicted_label=predicted_label,
        confidence=confidence,
        reasoning=reasoning[:300],
        is_correct=is_correct,
        raw_response=raw_vote,
    )


def choose_majority_label(
    votes: Sequence[JudgeVote],
    original_label: str,
) -> tuple[str, int, float, str]:
    valid_votes = [vote for vote in votes if vote.predicted_label != "ERROR"]
    if not valid_votes:
        return "ERROR", 0, 0.0, "low"

    counts = Counter(vote.predicted_label for vote in valid_votes)
    confidence_totals: dict[str, int] = {}
    for vote in valid_votes:
        confidence_totals[vote.predicted_label] = (
            confidence_totals.get(vote.predicted_label, 0)
            + CONFIDENCE_SCORES[vote.confidence]
        )

    winner_count = max(counts.values())
    tied_labels = [label for label, count in counts.items() if count == winner_count]
    if len(tied_labels) == 1:
        winner = tied_labels[0]
    elif original_label in tied_labels:
        winner = original_label
    else:
        winner = sorted(
            tied_labels,
            key=lambda label: (confidence_totals.get(label, 0), label),
            reverse=True,
        )[0]

    vote_fraction = winner_count / len(valid_votes)
    avg_confidence = confidence_totals.get(winner, 0) / max(1, winner_count)
    if len(tied_labels) > 1:
        ensemble_confidence = "low"
    elif vote_fraction >= HIGH_CONFIDENCE_FRACTION and avg_confidence >= 1.0:
        ensemble_confidence = "high"
    elif vote_fraction > MEDIUM_CONFIDENCE_FRACTION:
        ensemble_confidence = "medium"
    else:
        ensemble_confidence = "low"

    return winner, winner_count, vote_fraction, ensemble_confidence


def verify_single_example(
    index: int,
    task: str,
    source_split: str,
    source_index: int,
    labels: Sequence[str],
    text: str,
    original_label: str,
    judges: Sequence[JudgeConfig],
    args: argparse.Namespace,
) -> VerificationResult:
    prompt = create_verification_prompt(
        task=task,
        labels=labels,
        text=text,
        original_label=original_label,
        max_chars=args.max_chars,
    )

    judge_votes: list[JudgeVote] = []
    for judge in judges:
        raw_vote, error = call_judge(
            judge=judge,
            prompt=prompt,
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
        )
        judge_votes.append(normalize_vote(raw_vote, error, labels, judge))

    predicted_label, vote_count, vote_fraction, confidence = choose_majority_label(
        judge_votes,
        original_label=original_label,
    )
    is_correct = predicted_label == original_label
    suggested_correction = (
        None if is_correct or predicted_label == "ERROR" else predicted_label
    )

    return VerificationResult(
        index=index,
        task=task,
        source_split=source_split,
        source_index=source_index,
        text=text[:300],
        original_label=original_label,
        predicted_label=predicted_label,
        confidence=confidence,
        is_correct=is_correct,
        suggested_correction=suggested_correction,
        vote_count=vote_count,
        total_votes=len(
            [vote for vote in judge_votes if vote.predicted_label != "ERROR"]
        ),
        vote_fraction=vote_fraction,
        judge_votes=judge_votes,
    )


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def log_verify_progress(
    progress_label: str,
    completed: int,
    total: int,
    errors: int,
    started_at: float,
) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta = remaining / rate if rate > 0 else None
    logger.info(
        "%s progress: %s/%s (%.1f%%) rate=%.2f samples/s eta=%s errors=%s",
        progress_label,
        completed,
        total,
        completed / max(1, total) * 100,
        rate,
        format_eta(eta),
        errors,
    )


def verify_task_dataset(
    spec: TaskSpec,
    dataset: Dataset,
    indices: Sequence[int],
    judges: Sequence[JudgeConfig],
    args: argparse.Namespace,
    progress_label: str | None = None,
) -> list[VerificationResult]:
    futures = {}
    results: list[VerificationResult] = []
    progress_label = progress_label or spec.task

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index in indices:
            example = dataset[index]
            text = get_text(example, spec)
            original_label = normalize_label_name(example, spec)
            source_split = str(example.get(INTERNAL_SPLIT_COL, spec.split))
            source_index = int(example.get(INTERNAL_INDEX_COL, index))
            if not text or original_label is None:
                continue
            futures[
                executor.submit(
                    verify_single_example,
                    index,
                    spec.task,
                    source_split,
                    source_index,
                    spec.labels,
                    text,
                    original_label,
                    judges,
                    args,
                )
            ] = index

        total_futures = len(futures)
        if total_futures == 0:
            logger.warning("No valid samples to verify for %s", progress_label)
            return results

        progress_step = max(1, total_futures // 20)
        last_log_at = time.time()
        started_at = last_log_at
        completed = 0
        error_count = 0

        with tqdm(
            total=total_futures,
            desc=progress_label,
            unit="sample",
            dynamic_ncols=True,
        ) as progress_bar:
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    error_count += 1
                    logger.error("Verification error for %s: %s", progress_label, exc)
                else:
                    results.append(result)

                elapsed = max(time.time() - started_at, 1e-6)
                rate = completed / elapsed
                eta = (total_futures - completed) / rate if rate > 0 else None
                progress_bar.set_postfix_str(
                    f"errors={error_count} rate={rate:.2f}/s eta={format_eta(eta)}"
                )
                progress_bar.update(1)

                now = time.time()
                if (
                    completed == total_futures
                    or completed % progress_step == 0
                    or now - last_log_at >= PROGRESS_LOG_INTERVAL_SECONDS
                ):
                    log_verify_progress(
                        progress_label=progress_label,
                        completed=completed,
                        total=total_futures,
                        errors=error_count,
                        started_at=started_at,
                    )
                    last_log_at = now

    results.sort(key=lambda item: item.index)
    return results


def build_task_stats(
    spec: TaskSpec, results: Sequence[VerificationResult]
) -> TaskStats:
    stats = TaskStats()
    for label in spec.labels:
        stats.label_stats[label] = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "confused_with": {},
        }

    for result in results:
        stats.total += 1

        label_stats = stats.label_stats.get(result.original_label)
        if label_stats is not None:
            label_stats["total"] += 1

        if result.predicted_label == "ERROR" or result.total_votes == 0:
            stats.errors += 1
            continue

        if result.is_correct:
            stats.correct += 1
            if label_stats is not None:
                label_stats["correct"] += 1
            continue

        if result.confidence == "low":
            stats.uncertain += 1
            continue

        stats.incorrect += 1
        if label_stats is not None:
            label_stats["incorrect"] += 1
            confused_with = label_stats["confused_with"]
            confused_with[result.predicted_label] = (
                confused_with.get(result.predicted_label, 0) + 1
            )

    return stats


def should_escalate_stage1_result(
    result: VerificationResult,
    confidence_threshold: str,
) -> list[str]:
    reasons: list[str] = []

    if result.predicted_label == "ERROR" or result.total_votes == 0:
        reasons.append("stage1_error")
        return reasons

    if result.predicted_label != result.original_label:
        reasons.append("stage1_label_mismatch")

    if CONFIDENCE_SCORES[result.confidence] <= CONFIDENCE_SCORES[confidence_threshold]:
        reasons.append(f"stage1_{result.confidence}_confidence")

    stage1_vote = result.judge_votes[0] if result.judge_votes else None
    if (
        stage1_vote is not None
        and stage1_vote.is_correct is False
        and "stage1_label_mismatch" not in reasons
    ):
        reasons.append("stage1_marked_incorrect")

    return reasons


def merge_stage_results(
    stage1_results: Sequence[VerificationResult],
    stage2_results: Sequence[VerificationResult],
    stage1_escalation_threshold: str,
) -> tuple[list[VerificationResult], int]:
    stage2_by_index = {result.index: result for result in stage2_results}
    final_results: list[VerificationResult] = []
    escalated_count = 0

    for stage1_result in stage1_results:
        stage1_vote = (
            stage1_result.judge_votes[0] if stage1_result.judge_votes else None
        )
        stage1_result.stage1_vote = stage1_vote
        trigger_reasons = should_escalate_stage1_result(
            stage1_result,
            confidence_threshold=stage1_escalation_threshold,
        )
        stage1_result.stage2_trigger_reasons = trigger_reasons

        if not trigger_reasons:
            stage1_result.review_path = "stage1_only"
            final_results.append(stage1_result)
            continue

        escalated_count += 1
        stage2_result = stage2_by_index.get(stage1_result.index)
        if (
            stage2_result is None
            or stage2_result.predicted_label == "ERROR"
            or stage2_result.total_votes == 0
        ):
            if stage2_result is not None:
                stage1_result.stage2_judge_votes = stage2_result.judge_votes
            stage1_result.review_path = "stage1_fallback"
            final_results.append(stage1_result)
            continue

        stage2_result.stage1_vote = stage1_vote
        stage2_result.stage2_trigger_reasons = trigger_reasons
        stage2_result.review_path = "stage2"
        final_results.append(stage2_result)

    final_results.sort(key=lambda item: item.index)
    return final_results, escalated_count


def should_apply_correction(
    result: VerificationResult, confidence_threshold: str
) -> bool:
    if result.is_correct or not result.suggested_correction:
        return False
    return (
        CONFIDENCE_SCORES[result.confidence] >= CONFIDENCE_SCORES[confidence_threshold]
    )


def encode_label_value(spec: TaskSpec, new_label: str):
    if spec.label_storage == "name":
        return new_label
    return spec.labels.index(new_label)


def apply_corrections(
    dataset: Dataset,
    spec: TaskSpec,
    results: Sequence[VerificationResult],
    confidence_threshold: str,
) -> tuple[Dataset, int]:
    corrections = {
        result.index: result.suggested_correction
        for result in results
        if should_apply_correction(result, confidence_threshold)
    }
    logger.info("Applying %s corrections to %s", len(corrections), spec.task)

    def update_example(example, idx):
        new_label = corrections.get(idx)
        if new_label is None:
            return example

        original_label = normalize_label_name(example, spec)
        example["corrected"] = True
        example["original_label_name"] = original_label
        example["verified_label_name"] = new_label
        example[spec.label_col] = encode_label_value(spec, new_label)

        if spec.label_name_col and spec.label_name_col in example:
            example[spec.label_name_col] = new_label
        elif spec.task == "feedback":
            example["label_name"] = new_label

        return example

    corrected_dataset = dataset.map(
        update_example,
        with_indices=True,
        desc=f"Applying corrections for {spec.task}",
    )
    return corrected_dataset, len(corrections)


def save_corrected_dataset(
    dataset: Dataset,
    output_dir: Path,
    split_name: str,
    split_names: Sequence[str] | None = None,
) -> None:
    if split_name == "all-splits":
        split_order = list(split_names or [])
        if not split_order:
            split_order = sorted(set(dataset[INTERNAL_SPLIT_COL]))

        dataset_dict = DatasetDict()
        disk_path = output_dir / "corrected_all_splits"
        if disk_path.exists():
            shutil.rmtree(disk_path)

        for source_split in split_order:
            split_dataset = dataset.filter(
                lambda x, source_split=source_split: x[INTERNAL_SPLIT_COL]
                == source_split,
                desc=f"Collecting corrected {source_split}",
            )
            split_dataset = strip_internal_columns(split_dataset)
            dataset_dict[source_split] = split_dataset

            jsonl_path = output_dir / f"corrected_{source_split}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as handle:
                for example in split_dataset:
                    handle.write(json.dumps(example, ensure_ascii=False) + "\n")

        dataset_dict.save_to_disk(str(disk_path))
        return

    disk_path = output_dir / f"corrected_{split_name}"
    if disk_path.exists():
        shutil.rmtree(disk_path)
    dataset = strip_internal_columns(dataset)
    dataset.save_to_disk(str(disk_path))

    jsonl_path = output_dir / f"corrected_{split_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for example in dataset:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def strip_internal_columns(dataset: Dataset) -> Dataset:
    removable = [
        column
        for column in (INTERNAL_SPLIT_COL, INTERNAL_INDEX_COL)
        if column in dataset.column_names
    ]
    if removable:
        dataset = dataset.remove_columns(removable)
    return dataset


def print_task_report(
    task: str,
    stats: TaskStats,
    results: Sequence[VerificationResult],
    stage1_stats: TaskStats | None = None,
    stage2_stats: TaskStats | None = None,
    stage2_candidates: int = 0,
) -> None:
    print("\n" + "=" * 70)
    print(f"{task.upper()} DATASET VERIFICATION REPORT")
    print("=" * 70)
    print(f"Total verified: {stats.total}")
    print(
        f"Correct: {stats.correct} ({stats.correct / max(1, stats.total) * 100:.1f}%)"
    )
    print(
        f"Incorrect: {stats.incorrect} ({stats.incorrect / max(1, stats.total) * 100:.1f}%)"
    )
    print(
        f"Uncertain: {stats.uncertain} ({stats.uncertain / max(1, stats.total) * 100:.1f}%)"
    )
    print(f"Errors: {stats.errors}")

    if stage1_stats is not None:
        print("\nTwo-stage routing:")
        print(f"  Stage1 reviewed: {stage1_stats.total}")
        print(f"  Stage2 candidates: {stage2_candidates}")
        if stage2_stats is not None:
            print(f"  Stage2 reviewed: {stage2_stats.total}")

    review_path_counts = Counter(result.review_path for result in results)
    if review_path_counts:
        print("  Finalized by path:")
        for path, count in sorted(review_path_counts.items()):
            print(f"    {path}: {count}")

    print("\nPer-label:")
    for label, label_stats in stats.label_stats.items():
        total = label_stats["total"]
        if total == 0:
            continue
        print(
            f"  {label}: total={total}, correct={label_stats['correct']}, incorrect={label_stats['incorrect']}"
        )
        confused_with = label_stats["confused_with"]
        if confused_with:
            ordered_confusions = sorted(
                confused_with.items(), key=lambda item: -item[1]
            )
            top_confusions = ", ".join(
                f"{name}:{count}" for name, count in ordered_confusions[:5]
            )
            print(f"    confused_with={top_confusions}")

    bad_examples = [
        result
        for result in results
        if not result.is_correct
        and result.confidence != "low"
        and result.predicted_label != "ERROR"
    ]
    if bad_examples:
        print("\nSample voted corrections:")
        for result in bad_examples[:10]:
            print(
                f"  idx={result.index} original={result.original_label} voted={result.predicted_label} "
                f"split={result.source_split} source_idx={result.source_index} "
                f"confidence={result.confidence} votes={result.vote_count}/{max(1, result.total_votes)} "
                f"text={result.text[:120]!r}"
            )


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    if not args.stage1_model and not args.judge_model:
        raise SystemExit("At least one of --judge-model or --stage1-model is required.")

    stage1_judge = None
    if args.stage1_model:
        stage1_judges = build_judge_configs(
            model_names=[args.stage1_model],
            default_api_url=args.stage1_api_url or args.api_url,
            default_api_key=args.stage1_api_key or args.api_key,
            endpoint_entries=[],
        )
        stage1_judge = stage1_judges[0]

    judges = build_judge_configs(
        model_names=args.judge_model,
        default_api_url=args.api_url,
        default_api_key=args.api_key,
        endpoint_entries=args.judge_endpoint,
    )
    tasks = resolve_tasks(args.task)
    task_specs = build_task_specs(tasks, args.dataset_id_override)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Selected tasks: %s", tasks)
    logger.info(
        "Task dataset sources: %s",
        {task: task_specs[task].dataset_id for task in tasks},
    )
    if stage1_judge is not None:
        logger.info("Stage1 judge model: %s", stage1_judge.model)
    logger.info("Stage2/full-sweep judge models: %s", [judge.model for judge in judges])

    summary = {
        "tasks": {},
        "mode": "two_stage" if stage1_judge is not None else "one_stage",
        "stage1_judge": (
            serialize_judge_config(stage1_judge) if stage1_judge is not None else None
        ),
        "judges": [serialize_judge_config(judge) for judge in judges],
        "language": args.language,
        "sample": None if args.all else args.sample,
        "all": args.all,
        "confidence_threshold": args.confidence,
        "stage1_escalate_confidence": args.stage1_escalate_confidence,
    }

    for task in tasks:
        spec = task_specs[task]
        dataset, source_splits, requested_split = load_task_dataset(
            spec,
            split_override=args.split,
            language=args.language,
        )
        indices = choose_indices(len(dataset), args.sample, args.all, args.seed)
        logger.info(
            "Task %s: verifying %s/%s samples from split '%s'",
            task,
            len(indices),
            len(dataset),
            requested_split,
        )

        stage1_stats = None
        stage2_stats = None
        stage2_candidates = 0

        if stage1_judge is not None:
            stage1_results = verify_task_dataset(
                spec,
                dataset,
                indices,
                [stage1_judge],
                args,
                progress_label=f"{task} stage1",
            )
            stage1_stats = build_task_stats(spec, stage1_results)
            candidate_indices = [
                result.index
                for result in stage1_results
                if should_escalate_stage1_result(
                    result,
                    confidence_threshold=args.stage1_escalate_confidence,
                )
            ]
            stage2_candidates = len(candidate_indices)
            logger.info(
                "Task %s: stage2 candidates %s/%s",
                task,
                stage2_candidates,
                len(stage1_results),
            )

            stage2_results: list[VerificationResult] = []
            if judges and candidate_indices:
                stage2_results = verify_task_dataset(
                    spec,
                    dataset,
                    candidate_indices,
                    judges,
                    args,
                    progress_label=f"{task} stage2",
                )
                stage2_stats = build_task_stats(spec, stage2_results)

            results, stage2_candidates = merge_stage_results(
                stage1_results=stage1_results,
                stage2_results=stage2_results,
                stage1_escalation_threshold=args.stage1_escalate_confidence,
            )
        else:
            results = verify_task_dataset(
                spec,
                dataset,
                indices,
                judges,
                args,
                progress_label=task,
            )

        stats = build_task_stats(spec, results)
        print_task_report(
            task,
            stats,
            results,
            stage1_stats=stage1_stats,
            stage2_stats=stage2_stats,
            stage2_candidates=stage2_candidates,
        )

        task_dir = output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        report_path = task_dir / "verification_report.json"
        report_payload = {
            "task": task,
            "dataset_id": spec.dataset_id,
            "split": requested_split,
            "source_splits": source_splits,
            "language": args.language,
            "mode": summary["mode"],
            "stage1_stats": (
                stage1_stats.to_dict() if stage1_stats is not None else None
            ),
            "stage2_stats": (
                stage2_stats.to_dict() if stage2_stats is not None else None
            ),
            "stage2_candidates": stage2_candidates,
            "stats": stats.to_dict(),
            "results": [asdict(result) for result in results],
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, indent=2, ensure_ascii=False)
        logger.info("Saved report to %s", report_path)

        corrections_applied = 0
        if args.correct:
            corrected_dataset, corrections_applied = apply_corrections(
                dataset=dataset,
                spec=spec,
                results=results,
                confidence_threshold=args.confidence,
            )
            save_corrected_dataset(
                corrected_dataset,
                task_dir,
                requested_split,
                split_names=source_splits,
            )
            logger.info("Saved corrected dataset for %s", task)

        summary["tasks"][task] = {
            "stats": stats.to_dict(),
            "stage1_stats": (
                stage1_stats.to_dict() if stage1_stats is not None else None
            ),
            "stage2_stats": (
                stage2_stats.to_dict() if stage2_stats is not None else None
            ),
            "stage2_candidates": stage2_candidates,
            "report_path": str(report_path),
            "corrections_applied": corrections_applied,
        }

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
