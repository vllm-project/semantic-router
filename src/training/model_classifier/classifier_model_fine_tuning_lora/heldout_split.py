"""Reserve a held-out evaluation slice of MMLU-Pro before any training sampling.

MMLU-Pro publishes only 'validation' (70 rows) and 'test' (12032 rows), so the
intent classifier's training pool has to be carved out of 'test'. Everything a
held-out row asks therefore has to be taken off the gradient path up front, or
accuracy measured on 'test' is measuring rows the model trained on.

Kept free of torch, transformers, and sklearn so the contract tests can import it
under the stdlib-only python3 that `make test-training-contracts` runs.
See https://github.com/vllm-project/semantic-router/issues/3558
"""

from __future__ import annotations

import random

# Share of MMLU-Pro 'test' reserved for evaluation and never trained on.
HELDOUT_FRACTION = 0.2

# Fixed seed for every split, so a rerun reserves the same rows.
SPLIT_SEED = 42

MANIFEST_NAME = "heldout_eval.json"


def reserve_heldout(
    texts: list[str],
    labels: list[str],
    fraction: float = HELDOUT_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int]]:
    """Split row indices into a training pool and a reserved evaluation slice.

    The reserved slice takes ``fraction`` of each label, so every category keeps
    its share. MMLU-Pro repeats some question texts across rows, so reserving by
    row index alone would leave the same question on both sides; any pool row
    asking a reserved question is left out of the pool as well.

    Returns (pool_indices, heldout_indices), both sorted ascending. No question
    text appears in both.
    """
    if len(texts) != len(labels):
        raise ValueError(f"got {len(texts)} texts but {len(labels)} labels")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

    rows_by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        rows_by_label.setdefault(label, []).append(index)

    rng = random.Random(seed)
    heldout: list[int] = []
    for label in sorted(rows_by_label):
        rows = list(rows_by_label[label])
        rng.shuffle(rows)
        heldout.extend(rows[: round(len(rows) * fraction)])
    heldout.sort()

    reserved_rows = set(heldout)
    reserved_questions = {texts[i] for i in heldout}
    pool = [
        i
        for i in range(len(texts))
        if i not in reserved_rows and texts[i] not in reserved_questions
    ]
    return pool, heldout


def drop_reserved_questions(
    samples: list[tuple[str, str]], heldout_texts: list[str]
) -> list[tuple[str, str]]:
    """Drop (text, label) pairs asking a question the held-out slice already asks.

    The supplement dataset is merged into the training pool, so a supplement row
    repeating a reserved question would put it back on the gradient path.
    """
    reserved = set(heldout_texts)
    return [(text, label) for text, label in samples if text not in reserved]


def build_manifest(
    heldout_indices: list[int],
    metrics: dict[str, float],
    dataset: str,
    split: str = "test",
    fraction: float = HELDOUT_FRACTION,
    seed: int = SPLIT_SEED,
) -> dict:
    """Describe the reserved slice so the accuracy quoted for it is reproducible."""
    return {
        "dataset": dataset,
        "split": split,
        "heldout_fraction": fraction,
        "seed": seed,
        "num_heldout_rows": len(heldout_indices),
        "heldout_row_indices": list(heldout_indices),
        "accuracy": metrics["eval_accuracy"],
        "f1": metrics["eval_f1"],
    }
