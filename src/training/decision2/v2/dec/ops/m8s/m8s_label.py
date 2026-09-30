"""A20r (DEV2.0-27B) decision distributions for decoder M8-small top-up rows, on the scored runtime.

GPU, in the ~27B kernel image (container started by m8s-label.sh). `typed_collect_kernel.kernel_runtime()` runs
first (FLA overlay, kernel bindings, persisted TRITON_CACHE_DIR); the logits then come from the pinned
`training.model.calibrate.collect_logits`, the code of A20r's kernel-path CAL fit: FP32 parameters, BF16 backbone,
FP32 head, the LoRA checkpoint on its pinned base, no truncation at --max-length. T = 1: the package calibration must
bind this checkpoint with every temperature 1.0, and targets are softmax(raw logits).

  parity  re-collect CAL rows and compare with the stored CAL logits of the same checkpoint:
          PASS = 0 argmax changes and max |delta logit| <= 1e-4 (else FAIL; labeling must not start)
  label   rows of one or more TRAIN files, deduplicated by (id, input_sha256) and sorted by that key, shard
          --shard-index of --shard-count (position mod count) -> {id, input_sha256, teacher_probs} lines + manifest

usage: python3 v2/dec/ops/m8s/m8s_label.py parity|label --checkpoint C --source-path BASE --calibration CAL.json \
    --model-sha256 SHA [--max-length 32768] ...
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

LABEL_VERSION = "dec-m8s-a20r-labels/1"
PARITY_TOLERANCE = 1e-4
TEACHER_REPO = "llm-semantic-router/DEV2.0-27B"
TEACHER_REVISION = "5323310327e52d4eadd119cd10accac9b106c97d"


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = math.fsum(exps)
    return [v / total for v in exps]


def unique_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    from training.model.data import file_sha256, load_partition

    seen: dict[tuple[str, str], dict[str, Any]] = {}
    hashes = {}
    for path in paths:
        hashes[str(path)] = file_sha256(path)
        for row in load_partition(path, "train"):
            seen.setdefault((row["id"], row["input_sha256"]), row)
    return [seen[key] for key in sorted(seen)], hashes


def compare_logits(
    fresh: list[dict[str, Any]], stored: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {r["id"]: r for r in stored}
    if len(by_id) != len(stored) or {r["id"] for r in fresh} != set(by_id):
        raise ValueError("stored logits do not cover exactly the re-collected rows")
    max_delta = 0.0
    changes = exact = 0
    for record in fresh:
        old = by_id[record["id"]]["logits"]
        new = record["logits"]
        if len(old) != len(new):
            raise ValueError(f"{record['id']}: option count differs")
        max_delta = max(max_delta, max(abs(a - b) for a, b in zip(old, new)))
        exact += int(old == new)
        changes += int(
            max(range(len(old)), key=old.__getitem__)
            != max(range(len(new)), key=new.__getitem__)
        )
    status = "PASS" if changes == 0 and max_delta <= PARITY_TOLERANCE else "FAIL"
    return {
        "status": status,
        "rows": len(fresh),
        "argmax_changes": changes,
        "max_abs_logit_delta": max_delta,
        "bit_exact_rows": exact,
        "tolerance": PARITY_TOLERANCE,
    }


def teacher_identity(args: argparse.Namespace) -> dict[str, Any]:
    from training.model.calibration import load_calibration
    from training.model.infer import checkpoint_fingerprint

    identity = checkpoint_fingerprint(args.checkpoint, args.source_path)
    if identity["model_sha256"] != args.model_sha256:
        raise SystemExit(
            f"checkpoint identity {identity['model_sha256']} != expected {args.model_sha256}"
        )
    temperatures, _ = load_calibration(args.calibration, args.model_sha256)
    if any(float(t) != 1.0 for t in temperatures.values()):
        raise SystemExit(f"package calibration is not T = 1: {temperatures}")
    return {"model_sha256": identity["model_sha256"], "temperatures": temperatures}


def write_atomic(path: Path, lines: list[str]) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("x", encoding="utf-8") as stream:
        stream.writelines(lines)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=("parity", "label"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-path", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--cal", type=Path, help="parity: CAL rows")
    p.add_argument("--cal-sha256", help="parity: pinned CAL hash")
    p.add_argument(
        "--stored-logits", type=Path, help="parity: the checkpoint's stored CAL logits"
    )
    p.add_argument(
        "--train", type=Path, action="append", default=[], help="label: TRAIN file"
    )
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.mode == "parity" and not (a.cal and a.cal_sha256 and a.stored_logits):
        p.error("parity needs --cal, --cal-sha256 and --stored-logits")
    if a.mode == "label" and not a.train:
        p.error("label needs --train")
    if not 0 <= a.shard_index < a.shard_count:
        p.error("need 0 <= --shard-index < --shard-count")

    kernel = importlib.import_module("v2.27b.typed_collect_kernel")
    runtime = kernel.kernel_runtime()
    from training.model.calibrate import collect_logits
    from training.model.data import file_sha256, load_partition

    identity = teacher_identity(a)
    options = dict(
        source_path=a.source_path,
        max_length=a.max_length,
        batch_size=1,
        device_name="cuda:0",
    )
    started = time.perf_counter()
    if a.mode == "parity":
        if file_sha256(a.cal) != a.cal_sha256:
            raise SystemExit("CAL file differs from its pinned SHA-256")
        rows = load_partition(a.cal, "cal")
        fresh = collect_logits(a.checkpoint, rows, **options)
        stored = [json.loads(line) for line in a.stored_logits.open(encoding="utf-8")]
        result = compare_logits(fresh, stored)
        result.update(
            {
                "schema": "dec-m8s-a20r-parity/1",
                "teacher": identity,
                "cal_sha256": a.cal_sha256,
                "stored_logits_sha256": file_sha256(a.stored_logits),
                "seconds": time.perf_counter() - started,
                "runtime": runtime,
                "versions": kernel.versions(),
                "memory": kernel.memory(),
            }
        )
        a.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("status", "argmax_changes", "max_abs_logit_delta")
                }
            )
        )
        return 0 if result["status"] == "PASS" else 1

    rows, train_hashes = unique_rows(a.train)
    shard = [r for i, r in enumerate(rows) if i % a.shard_count == a.shard_index]
    records = collect_logits(a.checkpoint, shard, **options)
    if [r["id"] for r in records] != [r["id"] for r in shard]:
        raise SystemExit("logit collection dropped or reordered rows")
    agreement: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    lines = []
    for row, record in zip(shard, records):
        raw = record["logits"]
        if not all(math.isfinite(v) for v in raw):
            raise SystemExit(f"{row['id']}: nonfinite teacher logit")
        probs = softmax(raw)
        keys = [option["key"] for option in row["options"]]
        if len(keys) != len(probs):
            raise SystemExit(f"{row['id']}: option count differs from the logits")
        stats = agreement[row["task_type"]]
        stats[0] += int(max(range(len(probs)), key=probs.__getitem__) == row["label"])
        stats[1] += 1
        lines.append(
            json.dumps(
                {
                    "id": row["id"],
                    "input_sha256": row["input_sha256"],
                    "teacher_probs": dict(zip(keys, probs)),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
    write_atomic(a.output, lines)
    manifest = {
        "label_version": LABEL_VERSION,
        "teacher_repo": TEACHER_REPO,
        "teacher_revision": TEACHER_REVISION,
        "teacher": identity,
        "checkpoint": str(a.checkpoint),
        "source_path": str(a.source_path),
        "calibration_sha256": file_sha256(a.calibration),
        "precision": "FP32 parameters; BF16 backbone compute and FP32 head; batch size 1",
        "max_length": a.max_length,
        "train_sha256": train_hashes,
        "unique_rows": len(rows),
        "shard": {"index": a.shard_index, "count": a.shard_count, "rows": len(shard)},
        "output_sha256": file_sha256(a.output),
        "train_label_agreement": {
            kind: {"correct": c, "n": n, "accuracy": c / n}
            for kind, (c, n) in sorted(agreement.items())
        },
        "seconds": time.perf_counter() - started,
        "runtime": runtime,
        "versions": kernel.versions(),
        "memory": kernel.memory(),
    }
    a.output.with_name(a.output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps({"rows": len(shard), "seconds": manifest["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
