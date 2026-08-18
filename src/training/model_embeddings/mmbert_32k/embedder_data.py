"""BGE-style dataset loading helpers for the mmBERT embedder trainer."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 41e60e17ca960718dd1b71a23a86992128b9ed61.

from __future__ import annotations

import glob
import json
import logging
import os
from collections import defaultdict

from datasets import Dataset as HFDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_batch_size_for_length(length: int, base_batch_size: int = 32) -> int:
    """Return the historical BGE batch size for a sequence length."""
    mapping = [
        (7000, 0.25),
        (6000, 0.33),
        (5000, 0.4),
        (4000, 0.5),
        (3000, 0.6),
        (2000, 0.75),
        (1000, 1.0),
        (500, 1.5),
        (0, 2.0),
    ]
    for min_len, multiplier in mapping:
        if length >= min_len:
            return max(1, int(base_batch_size * multiplier))
    return base_batch_size


def load_bge_jsonl_file(filepath: str, max_samples: int | None = None) -> list[dict]:
    """Load valid query/positive records from one BGE-format JSONL file."""
    samples = []
    with open(filepath, encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if item.get("query") and item.get("pos"):
                samples.append(item)
                if max_samples and len(samples) >= max_samples:
                    break
    return samples


def _matches_language(filepath: str, languages: list[str] | None) -> bool:
    if not languages:
        return True
    filename = os.path.basename(filepath).lower()
    parent = os.path.basename(os.path.dirname(filepath)).lower()
    return any(lang in filename or lang in parent for lang in languages)


def _length_bucket(filepath: str) -> str:
    filename = os.path.basename(filepath)
    if "len-0-500" in filename:
        return "short"
    if "len-500-1000" in filename:
        return "medium"
    if "len-1000-2000" in filename:
        return "medium-long"
    if "len-" in filename:
        return "long"
    return "unknown"


def load_bge_data_directory(
    data_dir: str,
    max_samples_per_file: int | None = None,
    max_total_samples: int | None = None,
    languages: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Load BGE JSONL files grouped by their historical length buckets."""
    logger.info("Loading BGE data from: %s", data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True))
    logger.info("Found %s JSONL files", len(files))
    data_by_length: dict[str, list[dict]] = defaultdict(list)
    total_loaded = 0
    for filepath in tqdm(files, desc="Loading data files"):
        if not _matches_language(filepath, languages):
            continue
        samples = load_bge_jsonl_file(filepath, max_samples_per_file)
        data_by_length[_length_bucket(filepath)].extend(samples)
        total_loaded += len(samples)
        if max_total_samples and total_loaded >= max_total_samples:
            break
    logger.info("Loaded %s total samples", f"{total_loaded:,}")
    for bucket, samples in data_by_length.items():
        logger.info("  %s: %s samples", bucket, f"{len(samples):,}")
    return data_by_length


def convert_bge_to_triplets(
    data_by_length: dict[str, list[dict]], train_group_size: int = 2
) -> HFDataset:
    """Convert BGE records to sentence-transformers triplets."""
    del train_group_size  # Retained for compatibility with the upstream launcher API.
    triplets = []
    for samples in data_by_length.values():
        for item in samples:
            positives = item["pos"]
            negatives = item.get("neg", [])
            if not positives:
                continue
            positive = positives[0] if isinstance(positives, list) else positives
            if negatives:
                negative = negatives[0] if isinstance(negatives, list) else negatives
            else:
                negative = ""
            triplets.append(
                {
                    "anchor": item["query"],
                    "positive": positive,
                    "negative": negative,
                }
            )
    logger.info("Created %s triplets", f"{len(triplets):,}")
    return HFDataset.from_list(triplets)
