"""Dataset map transforms used by the legacy foundation-data paths."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 7feef3045b9a733d14228436b6c78993eb402a3e.

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def retokenize_dataset(
    dataset,
    source_tokenizer,
    target_tokenizer,
    max_length: int,
    num_proc: int = 4,
    *,
    pad_to_max_length: bool = True,
):
    """Decode a tokenized dataset and encode it with the target tokenizer."""
    logger.info("Re-tokenizing dataset...")
    logger.info("  Source vocab size: %s", source_tokenizer.vocab_size)
    logger.info("  Target vocab size: %s", target_tokenizer.vocab_size)

    def decode_and_retokenize(examples):
        texts = source_tokenizer.batch_decode(
            examples["input_ids"], skip_special_tokens=True
        )
        encodings = target_tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length" if pad_to_max_length else False,
            return_tensors=None,
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    return dataset.map(
        decode_and_retokenize,
        batched=True,
        batch_size=100,
        num_proc=num_proc,
        remove_columns=[
            column
            for column in dataset.column_names
            if column not in ["input_ids", "attention_mask"]
        ],
        desc="Re-tokenizing",
    )


def tokenize_dataset(
    dataset,
    tokenizer,
    max_length: int,
    text_column: str,
    num_proc: int = 4,
    *,
    pad_to_max_length: bool = True,
):
    """Tokenize a text dataset for masked-language-model training."""
    logger.info("Tokenizing dataset (text column: %s)...", text_column)

    def process_batch(examples):
        texts = examples[text_column]
        languages = examples.get("language", ["unknown"] * len(texts))
        encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length" if pad_to_max_length else False,
            return_tensors=None,
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "language": languages,
        }

    return dataset.map(
        process_batch,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=[
            column
            for column in dataset.column_names
            if column not in ["input_ids", "attention_mask", "language"]
        ],
        desc="Tokenizing",
    )
