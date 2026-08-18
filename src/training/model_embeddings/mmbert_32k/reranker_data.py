"""BGE, Quora, and FEVER loading for 2D Matryoshka reranker training."""
# Derived from Model-training@3bc41e1322ee5a53e08d18eb940855dec53c1539.
# Upstream blob: 36951a954bf2be62ea4d6536fecfd3ce0aad6d5c.

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
_MINIMUM_PAIR_SIZE = 2


@dataclass
class RerankerExample:
    """A query, one positive passage, and optional negatives."""

    query: str
    positive: str
    negatives: list[str] = field(default_factory=list)


class RerankerDataset(Dataset):
    """Dataset supporting the historical BGE, Quora, and FEVER sources."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        max_samples: int | None = None,
        negatives_per_query: int = 7,
        use_quora: bool = False,
        use_fever: bool = False,
        max_quora_samples: int | None = 100000,
        max_fever_samples: int | None = 100000,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.negatives_per_query = negatives_per_query
        self.examples: list[RerankerExample] = []
        if data_path:
            logger.info("Loading primary data from %s", data_path)
            self._load_primary(data_path, max_samples)
        logger.info("Loaded %s examples from primary data", len(self.examples))
        if use_quora:
            self._load_quora(max_quora_samples)
        if use_fever:
            self._load_fever(max_fever_samples)
        logger.info("Total examples: %s", len(self.examples))

    def _load_primary(self, data_path: str, max_samples: int | None) -> None:
        if os.path.isdir(data_path):
            self._load_directory(data_path, max_samples)
        elif os.path.exists(data_path):
            self._load_jsonl(data_path, max_samples)
        else:
            self._load_huggingface(data_path, max_samples)

    def _load_directory(self, directory: str, max_samples: int | None) -> None:
        files = glob.glob(os.path.join(directory, "**/*.jsonl"), recursive=True)
        logger.info("Found %s JSONL files", len(files))
        for filepath in sorted(files):
            if max_samples and len(self.examples) >= max_samples:
                break
            self._load_jsonl(filepath, max_samples)

    def _load_jsonl(self, filepath: str, max_samples: int | None) -> None:
        count = 0
        with open(filepath, encoding="utf-8") as handle:
            for line in handle:
                if max_samples and len(self.examples) >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                self._add_example(item)
                count += 1
        if count > 0:
            logger.debug("  Loaded %s from %s", count, os.path.basename(filepath))

    def _load_huggingface(self, dataset_name: str, max_samples: int | None) -> None:
        try:
            from datasets import load_dataset  # noqa: PLC0415

            logger.info("Loading from HuggingFace: %s", dataset_name)
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            count = 0
            for item in tqdm(dataset, desc=f"Loading {dataset_name}"):
                if max_samples and count >= max_samples:
                    break
                self._add_example(item)
                count += 1
            logger.info("Loaded %s examples from %s", count, dataset_name)
        except Exception as error:
            logger.warning("Failed to load %s: %s", dataset_name, error)

    def _load_quora(self, max_samples: int | None = 100000) -> None:
        try:
            from datasets import load_dataset  # noqa: PLC0415

            logger.info("Loading Quora Question Pairs...")
            dataset = load_dataset("quora", split="train")
            count = 0
            for item in tqdm(dataset, desc="Loading Quora"):
                if max_samples and count >= max_samples:
                    break
                questions = item.get("questions", {})
                texts = questions.get("text", []) if questions else []
                if len(texts) < _MINIMUM_PAIR_SIZE or not item.get("is_duplicate"):
                    continue
                self.examples.append(
                    RerankerExample(query=texts[0], positive=texts[1], negatives=[])
                )
                count += 1
            logger.info("Loaded %s examples from Quora", count)
        except Exception as error:
            logger.warning("Failed to load Quora: %s", error)

    def _load_primary_fever(self, load_dataset, max_samples: int | None) -> int:
        dataset = load_dataset("fever", "v1.0", split="train")
        count = 0
        for item in tqdm(dataset, desc="Loading FEVER"):
            if max_samples and count >= max_samples:
                break
            claim = item.get("claim", "")
            label = item.get("label", "")
            evidence = item.get("evidence_sentence", item.get("evidence", ""))
            if not claim or label != "SUPPORTS" or not evidence:
                continue
            positive = evidence if isinstance(evidence, str) else str(evidence)
            self.examples.append(
                RerankerExample(query=claim, positive=positive, negatives=[])
            )
            count += 1
        return count

    def _load_alternative_fever(self, load_dataset, max_samples: int | None) -> int:
        dataset = load_dataset("copenlu/fever_gold_evidence", split="train")
        count = 0
        for item in tqdm(dataset, desc="Loading FEVER"):
            if max_samples and count >= max_samples:
                break
            claim = item.get("claim", "")
            evidence = item.get("evidence", "")
            if not claim or not evidence or item.get("label", 0) != 0:
                continue
            self.examples.append(
                RerankerExample(query=claim, positive=evidence, negatives=[])
            )
            count += 1
        return count

    def _load_fever(self, max_samples: int | None = 100000) -> None:
        try:
            from datasets import load_dataset  # noqa: PLC0415

            logger.info("Loading FEVER dataset...")
            count = self._load_primary_fever(load_dataset, max_samples)
            logger.info("Loaded %s examples from FEVER", count)
        except Exception as error:
            logger.warning("Failed to load FEVER: %s", error)
            try:
                from datasets import load_dataset  # noqa: PLC0415

                logger.info("Trying alternative FEVER loading...")
                count = self._load_alternative_fever(load_dataset, max_samples)
                logger.info("Loaded %s examples from FEVER (alternative)", count)
            except Exception as alternative_error:
                logger.warning(
                    "Alternative FEVER loading also failed: %s", alternative_error
                )

    def _add_example(self, item: dict) -> None:
        query = item.get("query", item.get("question", ""))
        positives = item.get("pos", item.get("positive", []))
        negatives = item.get("neg", item.get("negative", []))
        if not query or not positives:
            return
        if isinstance(positives, str):
            positives = [positives]
        if isinstance(negatives, str):
            negatives = [negatives]
        for positive in positives[:1]:
            self.examples.append(
                RerankerExample(
                    query=query,
                    positive=positive,
                    negatives=negatives[: self.negatives_per_query],
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        pairs = [(example.query, example.positive)]
        labels = [1.0]
        for negative in example.negatives:
            pairs.append((example.query, negative))
            labels.append(0.0)
        encoded = self.tokenizer(
            [[query, passage] for query, passage in pairs],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def collate_fn(batch):
    """Flatten each query's positive/negative pair dimension into the batch."""
    return {
        "input_ids": torch.cat([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.cat([item["attention_mask"] for item in batch], dim=0),
        "labels": torch.cat([item["labels"] for item in batch], dim=0),
    }
