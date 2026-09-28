"""Cached frozen features as GPU tensors for head training and readout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

TASK_TYPES = ("choice", "noul", "score")
REPRESENTATIONS = ("joint", "disaggregated")


def read_rows(folder: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]


class FeatureSet:
    """One input set at one layer and representation.

    ``table`` holds the vectors; ``candidate_index`` [N, Kmax] and
    ``query_index`` [N] point into it (-1 pads). For the disaggregated
    representation the candidate index is also the text identity used to
    build in-batch InfoNCE pools and the cross-request candidate cache.
    """

    def __init__(
        self,
        folder: Path,
        layer: int,
        representation: str,
        device: torch.device,
        pooling: str = "last",
    ):
        from safetensors.torch import load_file

        if representation not in REPRESENTATIONS:
            raise ValueError(representation)
        self.folder, self.layer, self.representation = (
            Path(folder),
            layer,
            representation,
        )
        self.rows = read_rows(self.folder)
        if representation == "joint":
            self.table = load_file(str(self.folder / "joint.safetensors"))[
                f"L{layer}"
            ].to(device)
            valid_key = "j_valid"
        else:
            self.table = load_file(str(self.folder / "texts.safetensors"))[
                f"{pooling}_L{layer}"
            ].to(device)
            valid_key = "d_valid"
        n = len(self.rows)
        width = max(len(row["keys"]) for row in self.rows)
        candidate = torch.full((n, width), -1, dtype=torch.long)
        query = torch.full((n,), -1, dtype=torch.long)
        level = torch.full((n, width), -1, dtype=torch.long)
        self.valid = torch.zeros(n, dtype=torch.bool)
        for i, row in enumerate(self.rows):
            k = len(row["keys"])
            if not row[valid_key]:
                continue
            self.valid[i] = True
            if representation == "joint":
                candidate[i, :k] = torch.arange(row["j_offset"], row["j_offset"] + k)
                query[i] = row["j_offset"] + k
            else:
                candidate[i, :k] = torch.tensor(row["d_candidate_indices"])
                query[i] = row["d_state_index"]
            if row["task_type"] == "score":
                level[i, :k] = torch.tensor([int(key) for key in row["keys"]])
        self.candidate_index = candidate.to(device)
        self.query_index = query.to(device)
        self.level_index = level.to(device)
        self.counts = torch.tensor(
            [len(row["keys"]) for row in self.rows], device=device
        )
        self.task_type = torch.tensor(
            [TASK_TYPES.index(row["task_type"]) for row in self.rows], device=device
        )
        self.labels = torch.tensor(
            [row["label"] if row["label"] is not None else -1 for row in self.rows],
            device=device,
        )
        self.valid = self.valid.to(device)
        self.device = device

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        indices = indices.to(self.device)
        counts = self.counts[indices]
        width = int(counts.max())
        index = self.candidate_index[indices, :width]
        mask = index >= 0
        candidates = self.table[index.clamp(min=0)] * mask[..., None]
        query = self.table[self.query_index[indices].clamp(min=0)]
        return {
            "rows": indices,
            "candidates": candidates,
            "candidate_mask": mask,
            "candidate_ids": index,
            "query": query,
            "query_ids": self.query_index[indices],
            "labels": self.labels[indices],
            "task_type": self.task_type[indices],
            "counts": counts,
            "levels": self.level_index[indices, :width],
        }


def sorted_levels(batch: dict[str, torch.Tensor], score_rows: torch.Tensor):
    """Score-row candidate vectors reordered by ordinal level, plus gold level."""
    levels = batch["levels"][score_rows]
    order = levels.masked_fill(levels < 0, 10_000).argsort(dim=-1)
    vectors = batch["candidates"][score_rows].gather(
        1, order[..., None].expand(-1, -1, batch["candidates"].shape[-1])
    )
    counts = batch["counts"][score_rows]
    mask = (
        torch.arange(levels.shape[1], device=levels.device)[None, :] < counts[:, None]
    )
    labels = batch["labels"][score_rows]
    gold_level = levels.gather(1, labels.clamp(min=0)[:, None]).squeeze(1)
    return vectors, mask, counts, gold_level, order
