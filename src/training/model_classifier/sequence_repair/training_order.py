"""Replay a complete, content-bound training order without changing optimization."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .data import file_receipts


def ordered_ids_sha256(ids):
    """Canonical identity preserves UTF-8 IDs, duplicates, and their full order."""
    return hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate training-order field")
        result[key] = value
    return result


@dataclass
class TrainingOrder:
    """A step can only complete once, with every planned example accounted for."""

    indices: tuple[int, ...]
    eligible_ids: tuple[str, ...]
    steps: int
    global_batch: int
    receipt: dict
    completed_steps: int = field(default=0, init=False)

    def indices_for_step(self, step):
        if (
            type(step) is not int
            or step != self.completed_steps + 1
            or step > self.steps
        ):
            raise ValueError("Training order requires the next unconsumed step")
        start = (step - 1) * self.global_batch
        return list(self.indices[start : start + self.global_batch])

    def record_step(self, step, microbatches):
        planned = self.indices_for_step(step)
        actual = [index for batch in microbatches for index in batch]
        if Counter(planned) != Counter(actual):
            raise ValueError("Microbatches dropped or duplicated planned examples")
        self.completed_steps = step
        return {
            "step": step,
            "planned_ids": [self.eligible_ids[index] for index in planned],
            "microbatch_ids": [
                [self.eligible_ids[index] for index in batch] for batch in microbatches
            ],
        }

    def finish(self):
        if self.completed_steps != self.steps:
            raise ValueError("Training ended before consuming the complete order")
        return {
            **self.receipt,
            "completed_steps": self.completed_steps,
            "completed_draws": self.completed_steps * self.global_batch,
        }


def load_training_order(path, rows, train_paths, *, steps, global_batch):
    """Validate exact source bytes, eligible IDs, budgets, and planned draws."""
    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=_unique_object)
    required = {
        "version",
        "train_files",
        "eligible_ids_sha256",
        "steps",
        "global_batch",
        "ids",
        "ids_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Training order has unknown or missing fields")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unsupported training-order version")
    for name, expected in (("steps", steps), ("global_batch", global_batch)):
        if (
            type(expected) is not int
            or expected <= 0
            or type(value[name]) is not int
            or value[name] != expected
        ):
            raise ValueError("Training order does not match the optimizer budget")
    if value["train_files"] != file_receipts(train_paths):
        raise ValueError("Training order does not match TRAIN file contents")
    eligible = [row.get("id") for row in rows]
    if (
        not eligible
        or any(not isinstance(key, str) or not key for key in eligible)
        or len(set(eligible)) != len(eligible)
    ):
        raise ValueError("Eligible TRAIN IDs must be nonempty unique strings")
    if value["eligible_ids_sha256"] != ordered_ids_sha256(eligible):
        raise ValueError("Training order does not match eligible TRAIN IDs")
    ids = value["ids"]
    if (
        not isinstance(ids, list)
        or len(ids) != steps * global_batch
        or any(not isinstance(key, str) or not key for key in ids)
    ):
        raise ValueError("Training order needs exactly steps times global-batch IDs")
    if value["ids_sha256"] != ordered_ids_sha256(ids):
        raise ValueError("Training-order ID sequence hash mismatch")
    positions = {key: index for index, key in enumerate(eligible)}
    if any(key not in positions for key in ids):
        raise ValueError("Training order references unknown or filtered TRAIN IDs")
    receipt = {
        "file": Path(path).name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        **{key: value[key] for key in sorted(required - {"ids"})},
        "draws": len(ids),
        "unique_rows": len(set(ids)),
    }
    return TrainingOrder(
        tuple(positions[key] for key in ids),
        tuple(eligible),
        steps,
        global_batch,
        receipt,
    )
