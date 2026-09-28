"""Data identity, System One conversion and SELECT metrics shared by 0.6B arms.

Every arm sees the same text: flattened rights-clean rows become System One
requests exactly as `training.data.build_kai06b_native_v1.convert` builds them,
and the pinned Kai bundle's `system_one_training_rows` turns a request into
native records. SELECT correctness mirrors `training.model.train.evaluate`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RIGHTS_CLEAN_V2 = {
    "train": (
        "rights_clean.train.jsonl",
        "61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755",
    ),
    "select": (
        "select.jsonl",
        "32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6",
    ),
    "cal": (
        "cal.jsonl",
        "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a",
    ),
    "manifest": (
        "rights_clean.manifest.json",
        "61aa883052759830c4ecf897b36c1062ad816c935a12db824c80abd1f80e9ee8",
    ),
}
MAX_INPUT_TOKENS = 8192
TIE_TOLERANCE = 1e-8


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL line")
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, value: Any, *, exclusive: bool = False) -> str:
    path = Path(path)
    if exclusive and path.exists():
        raise FileExistsError(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return file_sha256(path)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> str:
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return file_sha256(path)


def load_rights_clean(parent: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Verify the frozen rights-clean v2 bytes, schema and partition isolation."""
    from training.model.data import check_partition_isolation, load_partition

    parent = Path(parent)
    for role, (name, expected) in RIGHTS_CLEAN_V2.items():
        path = parent / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError(
                f"rights-clean v2 {role} bytes differ from the frozen hash"
            )
    splits = {
        role: load_partition(parent / RIGHTS_CLEAN_V2[role][0], role)
        for role in ("train", "select", "cal")
    }
    check_partition_isolation(splits)
    return splits


def import_bundle(bundle: str | Path) -> Path:
    """Put one pinned Kai/Lex bundle root first on sys.path and return it."""
    root = Path(bundle).resolve(strict=True)
    for name in ("decision_runtime", "decision_inference", "decision_finetune"):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve().parent != root / name:
            raise RuntimeError("A different Kai/Lex bundle runtime is already imported")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def native_records(
    rows: list[dict[str, Any]], bundle: str | Path
) -> list[dict[str, Any]]:
    """Flattened rows -> labeled native records through the published converter."""
    from training.data.build_kai06b_native_v1 import convert

    import_bundle(bundle)
    from decision_finetune.system_one import system_one_training_rows

    records = []
    for row in rows:
        record, _ = convert(row, system_one_training_rows)
        record["source_row_id"] = row["id"]
        records.append(record)
    return records


def native_keys(record: dict[str, Any]) -> list[str]:
    question = record["question"]
    kind = question["type"].lower()
    if kind == "noul":
        return ["no", "yes"]
    return [
        option["id"] for option in question["options" if kind == "choice" else "levels"]
    ]


def original_probabilities(
    row: dict[str, Any], native_ids: list[str], probabilities: list[float]
) -> list[float]:
    """Map native candidate probabilities back to the flattened row's option order."""
    keys = [option["key"] for option in row["options"]]
    if len(probabilities) != len(native_ids) or len(keys) != len(native_ids):
        raise ValueError(f"{row['id']}: candidate count mismatch")
    if row["task_type"] == "noul":
        by_key = {
            "false": probabilities[native_ids.index("no")],
            "true": probabilities[native_ids.index("yes")],
        }
        return [by_key[key] for key in keys]
    if row["task_type"] == "choice":
        if native_ids != keys:
            raise ValueError(f"{row['id']}: native Choice order changed")
        return list(probabilities)
    if native_ids != [str(i) for i in range(len(keys))]:
        raise ValueError(f"{row['id']}: native Score levels are not positional")
    return list(probabilities)


def select_record(row: dict[str, Any], probabilities: list[float]) -> dict[str, Any]:
    """One SELECT/CAL row outcome, with the exact tie rules of the Qwen trainer."""
    keys = [option["key"] for option in row["options"]]
    p = [float(v) for v in probabilities]
    if (
        len(p) != len(keys)
        or not all(math.isfinite(v) and v >= 0 for v in p)
        or abs(sum(p) - 1) > 1e-4
    ):
        raise ValueError(f"{row['id']}: invalid probability vector")
    target = row["label"]
    if row["task_type"] == "noul":
        p_true = p[keys.index("true")]
        chosen = (
            None if p_true == 0.5 else keys.index("true" if p_true > 0.5 else "false")
        )
    else:
        top = max(p)
        matches = [i for i, value in enumerate(p) if abs(value - top) <= TIE_TOLERANCE]
        chosen = matches[0] if len(matches) == 1 else None
    return {
        "id": row["id"],
        "family": row["family"],
        "task_type": row["task_type"],
        "language": row["language"],
        "gold": target,
        "chosen": chosen,
        "correct": chosen == target,
        "brier": sum((value - float(i == target)) ** 2 for i, value in enumerate(p))
        / 2,
        "nll": -math.log(max(p[target], 1e-12)),
    }


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Same aggregation as `training.model.train.metric_summary`, plus type slices."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)
        by_type[record["task_type"]].append(record)

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(subset),
            "correct": sum(row["correct"] for row in subset),
            "accuracy": sum(row["correct"] for row in subset) / len(subset),
            "brier": sum(row["brier"] for row in subset) / len(subset),
            "nll": sum(row["nll"] for row in subset) / len(subset),
        }

    families = {name: block(subset) for name, subset in sorted(by_family.items())}
    score_levels = sorted(
        {
            row["chosen"]
            for row in records
            if row["task_type"] == "score" and row["chosen"] is not None
        }
    )
    return {
        "n": len(records),
        "correct": sum(row["correct"] for row in records),
        "micro_accuracy": sum(row["correct"] for row in records) / len(records),
        "family_macro_accuracy": sum(value["accuracy"] for value in families.values())
        / len(families),
        "family_macro_brier": sum(value["brier"] for value in families.values())
        / len(families),
        "by_family": families,
        "by_type": {name: block(subset) for name, subset in sorted(by_type.items())},
        "score_levels_predicted": score_levels,
    }


def better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    """BEST rule: family-macro accuracy, then lower macro Brier, then earlier step."""
    if incumbent is None:
        return True
    a, b = candidate["metrics"], incumbent["metrics"]
    if a["family_macro_accuracy"] != b["family_macro_accuracy"]:
        return a["family_macro_accuracy"] > b["family_macro_accuracy"]
    if a["family_macro_brier"] != b["family_macro_brier"]:
        return a["family_macro_brier"] < b["family_macro_brier"]
    return candidate["step"] < incumbent["step"]


def schedule(ids: list[str], logical_batch: int, seed: str) -> list[list[int]]:
    """One epoch in a fixed hash order; the last logical batch may be partial."""
    order = sorted(
        range(len(ids)),
        key=lambda i: (
            hashlib.sha256(f"{seed}\0{ids[i]}".encode()).hexdigest(),
            ids[i],
        ),
    )
    return [order[i : i + logical_batch] for i in range(0, len(order), logical_batch)]


def learning_rate(
    step: int, total: int, warmup: int, base: float, minimum: float
) -> float:
    """Kai2's `decision_finetune.state.learning_rate`: warmup, then cosine to `minimum`."""
    if not (1 <= step <= total and 0 <= warmup < total and 0 <= minimum <= base):
        raise ValueError("Invalid learning-rate horizon")
    if warmup and step <= warmup:
        return base * step / warmup
    if warmup:
        progress = (step - warmup) / (total - warmup)
    else:
        progress = (step - 1) / (total - 1) if total > 1 else 0.0
    return minimum + (base - minimum) * 0.5 * (1 + math.cos(math.pi * progress))
