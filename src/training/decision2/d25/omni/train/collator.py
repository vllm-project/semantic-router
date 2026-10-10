"""Multimodal collator: text-only and image rows in one left-padded batch.

Each row renders with the checkpoint prompt (images first, one placeholder per image), images are
decoded and resized by the processor at the 1.6 MP cap, and the batch carries ``pixel_values`` and
``image_grid_thw`` for all images in placeholder order (absent when no row has images). Targets are
padded to the 255 codes; choice options can be shuffled per step (targets follow the permutation).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch

from d25.omni.common import vision_format
from d25.omni.model import checkpoint, inputs
from d25.omni.train.rows import TrainRow


@dataclass
class Batch:
    inputs: dict[str, torch.Tensor]
    targets: torch.Tensor
    counts: torch.Tensor
    weights: torch.Tensor
    ids: list[str]
    tokens: int
    images: int

    def to(self, device) -> "Batch":
        return replace(
            self,
            inputs={name: value.to(device) for name, value in self.inputs.items()},
            targets=self.targets.to(device),
            counts=self.counts.to(device),
            weights=self.weights.to(device),
        )


def shuffle_options(row: TrainRow, rng: random.Random) -> TrainRow:
    if row.question["type"] != "choice":
        return row
    items = list(zip(row.question["criteria"].items(), row.target))
    rng.shuffle(items)
    question = {**row.question, "criteria": dict(item for item, _ in items)}
    return replace(row, question=question, target=[value for _, value in items])


class MultimodalCollator:
    def __init__(
        self, processor, codes: Sequence[str], prompt: str = "d25-vega"
    ) -> None:
        self.processor = processor
        self.codes = list(codes)
        self.prompt = prompt

    def render(self, row: TrainRow) -> str:
        return inputs.render(
            self.processor,
            self.prompt,
            row.state,
            row.question,
            self.codes,
            len(row.images),
        )

    def __call__(self, rows: Sequence[TrainRow], dummy: bool = False) -> Batch:
        if not rows:
            raise ValueError("a batch needs at least one row")
        texts = [self.render(row) for row in rows]
        images = [vision_format.load_image(ref) for row in rows for ref in row.images]
        encoded = inputs.encode(self.processor, texts, images)
        planned = max(row.tokens for row in rows)
        if planned and encoded["input_ids"].shape[1] != planned:
            raise RuntimeError(
                f"planned {planned} tokens, processor produced {encoded['input_ids'].shape[1]}"
            )
        targets = torch.zeros(len(rows), checkpoint.NUM_CODES, dtype=torch.float32)
        for index, row in enumerate(rows):
            targets[index, : row.n_options] = torch.tensor(
                row.target, dtype=torch.float32
            )
        weights = torch.tensor(
            [0.0 if dummy else row.weight for row in rows], dtype=torch.float32
        )
        return Batch(
            inputs=dict(encoded),
            targets=targets,
            counts=torch.tensor([row.n_options for row in rows], dtype=torch.long),
            weights=weights,
            ids=[row.id for row in rows],
            tokens=int(encoded["attention_mask"].sum()),
            images=len(images),
        )
