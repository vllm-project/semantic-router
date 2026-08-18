"""Hugging Face dataset preparation for the mmBERT foundation recipe."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 7feef3045b9a733d14228436b6c78993eb402a3e.

from __future__ import annotations

import logging
import os

from .foundation_data_transforms import retokenize_dataset, tokenize_dataset
from .foundation_integrity import load_source_prefix_contract
from .foundation_materialize import prepare_streaming_packed_dataset

logger = logging.getLogger(__name__)

_MULTILINGUAL_DATASETS = {
    "cc100": {"text_column": "text", "needs_lang": True},
    "statmt/cc100": {"text_column": "text", "needs_lang": True},
    "mc4": {"text_column": "text", "needs_lang": True},
    "oscar": {"text_column": "text", "needs_lang": True},
    "wikipedia": {
        "text_column": "text",
        "needs_lang": True,
        "date": "20231101",
    },
}
_FALLBACK_SOURCE_TOKENIZERS = (
    "meta-llama/Llama-2-7b-hf",
    "answerdotai/ModernBERT-base",
    "gpt2",
)


def _prepare_exact_dataset(
    *,
    dataset_name: str,
    output_dir: str,
    max_length: int,
    model_name: str,
    model_revision: str | None,
    dataset_revision: str | None,
    languages: list[str] | None,
    target_sequence_count: int,
    source_etags: list[str] | None,
    source_content_lengths: list[int] | None,
    source_prefix_contract_dir: str | None,
    max_document_bytes: int,
    max_document_tokens: int,
    acknowledge_cc100_license_unknown: bool,
):
    exact_languages = languages or []
    exact_etags = source_etags or []
    exact_lengths = source_content_lengths or []
    expected_prefixes = None
    if source_prefix_contract_dir:
        expected_prefixes = load_source_prefix_contract(
            source_prefix_contract_dir,
            languages=exact_languages,
            source_etags=exact_etags,
            source_content_lengths=exact_lengths,
        )
    return prepare_streaming_packed_dataset(
        dataset_name=dataset_name,
        dataset_revision=dataset_revision or "",
        output_dir=output_dir,
        languages=exact_languages,
        target_sequence_count=target_sequence_count,
        max_length=max_length,
        model_name=model_name,
        model_revision=model_revision or "",
        source_etags=exact_etags,
        source_content_lengths=exact_lengths,
        max_document_bytes=max_document_bytes,
        max_document_tokens=max_document_tokens,
        acknowledge_cc100_license_unknown=acknowledge_cc100_license_unknown,
        expected_source_prefixes=expected_prefixes,
    )


def _load_language_dataset(load_dataset, dataset_name, config, language, revision):
    common = {"split": "train", "streaming": False, "revision": revision}
    if dataset_name in {"cc100", "statmt/cc100"}:
        return load_dataset(dataset_name, language, **common)
    if dataset_name == "mc4":
        return load_dataset("mc4", language, **common)
    if dataset_name == "oscar":
        return load_dataset(
            "oscar-corpus/OSCAR-2301",
            language,
            trust_remote_code=True,
            **common,
        )
    if dataset_name == "wikipedia":
        return load_dataset(
            "wikipedia",
            f"{config['date']}.{language}",
            split="train",
            revision=revision,
        )
    return load_dataset(dataset_name, language, **common)


def _load_multilingual_dataset(
    load_dataset,
    concatenate_datasets,
    dataset_name: str,
    languages: list[str] | None,
    max_samples: int | None,
    dataset_revision: str | None,
):
    config = _MULTILINGUAL_DATASETS[dataset_name]
    selected_languages = languages or ["en"]
    logger.info("Loading multilingual dataset with languages: %s", selected_languages)
    samples_per_language = (
        max_samples // len(selected_languages) if max_samples else None
    )
    datasets = []
    for language in selected_languages:
        try:
            logger.info("  Loading %s...", language)
            dataset = _load_language_dataset(
                load_dataset, dataset_name, config, language, dataset_revision
            )
            if samples_per_language:
                dataset = dataset.select(range(min(samples_per_language, len(dataset))))
            dataset = dataset.add_column("language", [language] * len(dataset))
            datasets.append(dataset)
            logger.info("    Loaded %s samples for %s", len(dataset), language)
        except Exception as error:  # Dataset builders fail independently by language.
            logger.warning("  Could not load %s: %s", language, error)
    if not datasets:
        raise ValueError(f"Could not load any languages from {dataset_name}")
    combined = concatenate_datasets(datasets)
    logger.info("Combined dataset size: %s examples", len(combined))
    return combined, config["text_column"]


def _load_source_tokenizer(tokenizer_type, name: str | None, revision: str | None):
    if name:
        tokenizer = tokenizer_type.from_pretrained(name, revision=revision)
        logger.info("  Decoding with source tokenizer: %s", name)
        return tokenizer
    for candidate in _FALLBACK_SOURCE_TOKENIZERS:
        try:
            tokenizer = tokenizer_type.from_pretrained(candidate)
            logger.info("  Using source tokenizer: %s", candidate)
            return tokenizer
        except Exception:
            continue
    raise RuntimeError("none of the historical fallback source tokenizers loaded")


def _tokenize_text_dataset(
    dataset,
    tokenizer_type,
    model_name: str,
    model_revision: str | None,
    max_length: int,
    text_column: str,
    num_proc: int,
    pad_to_max_length: bool,
):
    logger.info("Tokenizing with %s...", model_name)
    tokenizer = tokenizer_type.from_pretrained(model_name, revision=model_revision)
    tokenizer.model_max_length = max_length
    return tokenize_dataset(
        dataset,
        tokenizer,
        max_length,
        text_column,
        num_proc,
        pad_to_max_length=pad_to_max_length,
    )


def _prepare_standard_dataset(
    dataset,
    tokenizer_type,
    *,
    model_name: str,
    model_revision: str | None,
    max_length: int,
    num_proc: int,
    force_retokenize: bool,
    source_tokenizer_name: str | None,
    source_tokenizer_revision: str | None,
    pad_to_max_length: bool,
):
    if "input_ids" not in dataset.column_names:
        text_column = (
            "text" if "text" in dataset.column_names else dataset.column_names[0]
        )
        return _tokenize_text_dataset(
            dataset,
            tokenizer_type,
            model_name,
            model_revision,
            max_length,
            text_column,
            num_proc,
            pad_to_max_length,
        )
    if not force_retokenize:
        return dataset
    logger.info("Re-tokenizing dataset with %s...", model_name)
    source = _load_source_tokenizer(
        tokenizer_type, source_tokenizer_name, source_tokenizer_revision
    )
    target = tokenizer_type.from_pretrained(model_name, revision=model_revision)
    target.model_max_length = max_length
    return retokenize_dataset(
        dataset,
        source,
        target,
        max_length,
        num_proc,
        pad_to_max_length=pad_to_max_length,
    )


def _save_dataset(dataset, output_dir: str) -> None:
    if "input_ids" in dataset.column_names:
        lengths = [len(value) for value in dataset["input_ids"][:100]]
        logger.info(
            "Token lengths (first 100): min=%s, max=%s", min(lengths), max(lengths)
        )
    logger.info("Saving dataset to: %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    logger.info("Dataset prepared successfully!")


def download_from_huggingface(
    dataset_name: str,
    output_dir: str,
    max_samples: int | None = None,
    num_proc: int = 4,
    languages: list[str] | None = None,
    max_length: int = 32768,
    model_name: str = "jhu-clsp/mmBERT-base",
    model_revision: str | None = None,
    dataset_revision: str | None = None,
    force_retokenize: bool = False,
    source_tokenizer_name: str | None = None,
    source_tokenizer_revision: str | None = None,
    target_sequence_count: int | None = None,
    source_etags: list[str] | None = None,
    source_content_lengths: list[int] | None = None,
    source_prefix_contract_dir: str | None = None,
    max_document_bytes: int = 8 * 1024 * 1024,
    max_document_tokens: int = 1024 * 1024,
    acknowledge_cc100_license_unknown: bool = False,
    pad_to_max_length: bool = True,
):
    """Download and prepare a revision-pinned Hugging Face dataset."""
    from datasets import concatenate_datasets, load_dataset  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    logger.info("Downloading dataset: %s", dataset_name)
    if target_sequence_count is not None:
        if max_samples is not None:
            raise ValueError(
                "max_samples cannot be combined with an exact target_sequence_count"
            )
        if force_retokenize or source_tokenizer_name is not None:
            raise ValueError("exact raw-text packing cannot re-tokenize token IDs")
        return _prepare_exact_dataset(
            dataset_name=dataset_name,
            output_dir=output_dir,
            max_length=max_length,
            model_name=model_name,
            model_revision=model_revision,
            dataset_revision=dataset_revision,
            languages=languages,
            target_sequence_count=target_sequence_count,
            source_etags=source_etags,
            source_content_lengths=source_content_lengths,
            source_prefix_contract_dir=source_prefix_contract_dir,
            max_document_bytes=max_document_bytes,
            max_document_tokens=max_document_tokens,
            acknowledge_cc100_license_unknown=acknowledge_cc100_license_unknown,
        )
    if dataset_name in _MULTILINGUAL_DATASETS:
        dataset, text_column = _load_multilingual_dataset(
            load_dataset,
            concatenate_datasets,
            dataset_name,
            languages,
            max_samples,
            dataset_revision,
        )
        if "input_ids" not in dataset.column_names:
            dataset = _tokenize_text_dataset(
                dataset,
                AutoTokenizer,
                model_name,
                model_revision,
                max_length,
                text_column,
                num_proc,
                pad_to_max_length,
            )
    else:
        dataset = load_dataset(dataset_name, split="train", revision=dataset_revision)
        logger.info("Dataset size: %s examples", len(dataset))
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info("Truncated to %s samples", len(dataset))
        dataset = _prepare_standard_dataset(
            dataset,
            AutoTokenizer,
            model_name=model_name,
            model_revision=model_revision,
            max_length=max_length,
            num_proc=num_proc,
            force_retokenize=force_retokenize,
            source_tokenizer_name=source_tokenizer_name,
            source_tokenizer_revision=source_tokenizer_revision,
            pad_to_max_length=pad_to_max_length,
        )
    _save_dataset(dataset, output_dir)
    return dataset
