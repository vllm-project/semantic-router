"""Classifier scorers for the red-team benchmark.

Only the probability of the jailbreak class is exposed, which is what the
router itself thresholds and all a black-box attacker gets to see.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

JAILBREAK_LABEL = "jailbreak"
BINARY_LABEL_COUNT = 2


def _jailbreak_index(model_dir: Path, config: Any) -> int:
    """Locate the jailbreak logit, preferring the label map the artifact ships."""
    mapping_path = model_dir / "label_mapping.json"
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text())
        labels = mapping.get("label_to_idx") or mapping.get("label2id") or mapping
        if isinstance(labels, dict):
            for label, index in labels.items():
                if str(label).lower() == JAILBREAK_LABEL:
                    return int(index)
    id2label = getattr(config, "id2label", None) or {}
    for index, label in id2label.items():
        if str(label).lower() == JAILBREAK_LABEL:
            return int(index)
    if getattr(config, "num_labels", BINARY_LABEL_COUNT) == BINARY_LABEL_COUNT:
        return 1
    raise ValueError(f"cannot locate a '{JAILBREAK_LABEL}' label in {model_dir}")


class TransformerScorer:
    """Scores prompts with a local sequence-classification artifact."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=torch.float32
        )
        self._model.eval().to(self._device)
        config = AutoConfig.from_pretrained(model_path)
        self._index = _jailbreak_index(Path(model_path), config)

    @property
    def device(self) -> str:
        return self._device

    def score(self, texts: Sequence[str]) -> list[float]:
        torch = self._torch
        scores: list[float] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1)[:, self._index]
            scores.extend(probabilities.cpu().tolist())
        return scores
