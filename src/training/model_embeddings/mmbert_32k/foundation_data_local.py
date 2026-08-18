"""Local-corpus and long-context helpers for foundation-data preparation."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 7feef3045b9a733d14228436b6c78993eb402a3e.

from __future__ import annotations

import logging
import os
from collections import Counter
from glob import glob

from .foundation_packing import real_token_ids

logger = logging.getLogger(__name__)
_NATIVE_CONTEXT_LENGTH = 8192


def _load_language_detector(enabled: bool):
    if not enabled:
        return None
    try:
        import langdetect  # noqa: PLC0415

        logger.info("Language detection enabled")
        return langdetect
    except ImportError:
        logger.warning("langdetect not installed. Install with: pip install langdetect")
        return None


def _detect_language(detector, text: str) -> str:
    if detector is None:
        return "unknown"
    try:
        return detector.detect(text[:1000])
    except Exception:
        return "unknown"


def _collect_text_files(patterns: list[str], detector):
    texts = []
    languages = []
    for pattern in patterns:
        for filepath in glob(pattern):
            logger.info("Reading: %s", filepath)
            with open(filepath, encoding="utf-8") as handle:
                text = handle.read()
            texts.append(text)
            languages.append(_detect_language(detector, text))
    if not texts:
        raise ValueError(f"No text files found matching: {patterns}")
    logger.info("Found %s text files", len(texts))
    if detector is not None:
        logger.info("Detected languages: %s", dict(Counter(languages)))
    return texts


def _chunk_local_tokens(tokens: list[int], tokenizer, max_length: int) -> list[dict]:
    examples = []
    for offset in range(0, len(tokens) - max_length + 1, max_length):
        chunk = tokens[offset : offset + max_length]
        input_ids = [tokenizer.cls_token_id, *chunk[:-1]]
        if tokenizer.sep_token_id:
            input_ids[-1] = tokenizer.sep_token_id
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "language": "mixed",
            }
        )
    return examples


def _examples_to_dataset(dataset_type, examples):
    return dataset_type.from_dict(
        {
            "input_ids": [example["input_ids"] for example in examples],
            "attention_mask": [example["attention_mask"] for example in examples],
            "language": [example["language"] for example in examples],
        }
    )


def tokenize_local_files(
    input_files: list[str],
    output_dir: str,
    max_length: int = 32768,
    model_name: str = "jhu-clsp/mmBERT-base",
    model_revision: str | None = None,
    max_samples: int | None = None,
    num_proc: int = 4,
    detect_language: bool = False,
):
    """Tokenize local text files using the historical full-corpus flow."""
    from datasets import Dataset  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    del num_proc  # Kept in the imported CLI contract; upstream did not use it here.
    logger.info("Loading tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    tokenizer.model_max_length = max_length
    detector = _load_language_detector(detect_language)
    texts = _collect_text_files(input_files, detector)
    logger.info("Tokenizing full corpus...")
    tokens = tokenizer.encode("\n\n".join(texts), add_special_tokens=False)
    logger.info("Total tokens: %s", f"{len(tokens):,}")
    examples = _chunk_local_tokens(tokens, tokenizer, max_length)
    logger.info("Created %s training examples", len(examples))
    if max_samples:
        examples = examples[:max_samples]
        logger.info("Truncated to %s samples", len(examples))
    dataset = _examples_to_dataset(Dataset, examples)
    logger.info("Saving dataset to: %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    logger.info("Dataset prepared successfully!")
    return dataset


def _real_sample_lengths(dataset, has_attention_mask: bool) -> list[int]:
    return [
        len(
            real_token_ids(
                row["input_ids"],
                row.get("attention_mask") if has_attention_mask else None,
            )
        )
        for row in dataset.select(range(min(100, len(dataset))))
    ]


def _pad_or_truncate(ids: list[int], target_length: int):
    output_ids = ids[:target_length]
    output_mask = [1] * len(output_ids)
    if len(output_ids) < target_length:
        padding = target_length - len(output_ids)
        output_ids.extend([0] * padding)
        output_mask.extend([0] * padding)
    return output_ids, output_mask


def concatenate_to_long_context(dataset, target_length: int = 32768, num_proc: int = 4):
    """Concatenate real attended tokens into fixed long-context sequences."""
    from datasets import Dataset  # noqa: PLC0415

    del num_proc  # Historical compatibility argument.
    logger.info("Concatenating sequences to target length: %s", target_length)
    has_attention_mask = "attention_mask" in dataset.column_names
    sample_lengths = _real_sample_lengths(dataset, has_attention_mask)
    if max(sample_lengths) >= target_length * 0.9:
        logger.info("Dataset already has long sequences, skipping concatenation")
        return dataset
    logger.info("  Original avg length: %s", sum(sample_lengths) // len(sample_lengths))
    concatenated_ids = []
    concatenated_masks = []
    current_ids: list[int] = []
    for row in dataset:
        ids = real_token_ids(
            row["input_ids"],
            row.get("attention_mask") if has_attention_mask else None,
        )
        if len(current_ids) + len(ids) <= target_length:
            current_ids.extend(ids)
            continue
        if len(current_ids) >= target_length // 2:
            output_ids, output_mask = _pad_or_truncate(current_ids, target_length)
            concatenated_ids.append(output_ids)
            concatenated_masks.append(output_mask)
        current_ids = list(ids)
    if len(current_ids) >= target_length // 2:
        output_ids, output_mask = _pad_or_truncate(current_ids, target_length)
        concatenated_ids.append(output_ids)
        concatenated_masks.append(output_mask)
    logger.info(
        "  Created %s sequences of length %s", len(concatenated_ids), target_length
    )
    return Dataset.from_dict(
        {"input_ids": concatenated_ids, "attention_mask": concatenated_masks}
    )


def verify_dataset(dataset_path: str) -> bool:
    """Verify that a prepared dataset contains genuinely long token sequences."""
    from datasets import load_from_disk  # noqa: PLC0415

    logger.info("Loading dataset from: %s", dataset_path)
    dataset = load_from_disk(dataset_path)
    logger.info("Dataset size: %s examples", len(dataset))
    logger.info("Columns: %s", dataset.column_names)
    if "input_ids" not in dataset.column_names:
        logger.error("Dataset missing 'input_ids' column!")
        return False
    lengths = [len(value) for value in dataset["input_ids"][:100]]
    logger.info("Token lengths (first 100):")
    logger.info("  Min: %s", min(lengths))
    logger.info("  Max: %s", max(lengths))
    logger.info("  Avg: %s", sum(lengths) // len(lengths))
    if max(lengths) <= _NATIVE_CONTEXT_LENGTH:
        logger.warning("WARNING: Dataset appears truncated at 8192 tokens!")
        logger.warning("Consider using --concatenate_to_length for 32K training.")
        return False
    logger.info("Dataset verification passed!")
    return True
