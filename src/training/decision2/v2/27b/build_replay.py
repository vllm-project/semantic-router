"""Teacher-target replay files for the ~27B KL arms (CPU, deterministic).

The KL arms keep the M2-C control's rows and tokens: TRAIN = pk1 A0s admitted at
the training limit, and the replay pool = the pk1 versions of exactly the
control's A0s-resample rows (ids suffixed ``#r1``), each carrying a teacher's
``teacher_probs``. With the trainer's replay fraction 1/3 every pool row is
used once, so the epoch has the control's 10,748 examples and 672 updates;
only the KL term on the resample block (and the pk1 option keys) differ.
Teacher rows join by ``id`` and must match the pk1 ``input_sha256`` and option
keys and sum to one within 1e-5.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from v2.eval.same_panel import sha_file

SUFFIX = "#r1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def attach(
    rows: list[dict[str, Any]], teacher: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out, correct = [], Counter()
    for row in rows:
        target = teacher.get(row["id"])
        if target is None:
            raise ValueError(f"{row['id']}: no teacher target")
        if target["input_sha256"] != row["input_sha256"]:
            raise ValueError(
                f"{row['id']}: teacher input hash differs from the pk1 row"
            )
        keys = [option["key"] for option in row["options"]]
        probs = target["teacher_probs"]
        if set(probs) != set(keys):
            raise ValueError(f"{row['id']}: teacher keys differ from the options")
        if (
            any(
                not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0
                for p in probs.values()
            )
            or abs(sum(probs.values()) - 1.0) > 1e-5
        ):
            raise ValueError(
                f"{row['id']}: teacher probabilities are not a distribution"
            )
        top = max(keys, key=lambda k: probs[k])
        correct[row["task_type"], top == keys[row["label"]]] += 1
        out.append(
            {
                **row,
                "id": row["id"] + SUFFIX,
                "teacher_probs": {k: probs[k] for k in keys},
            }
        )
    agreement = {
        kind: correct[kind, True] / (correct[kind, True] + correct[kind, False])
        for kind in sorted({k for k, _ in correct})
    }
    return out, {"rows": len(out), "teacher_argmax_equals_gold": agreement}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pk1-a0s", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True, help="M2-C mixture file")
    parser.add_argument(
        "--teacher", action="append", required=True, help="NAME=TARGETS"
    )
    parser.add_argument("--expect", action="append", default=[], help="PATH=SHA256")
    parser.add_argument(
        "--source", type=Path, required=True, help="Base snapshot (tokenizer)"
    )
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from training.model.data import load_partition, validate_row
    from training.model.decision_model import encode

    inputs = [args.pk1_a0s, args.control] + [
        Path(s.split("=", 1)[1]) for s in args.teacher
    ]
    expected = dict(item.rsplit("=", 1) for item in args.expect)
    hashes = {str(p): sha_file(p) for p in inputs}
    for path, digest in hashes.items():
        if expected.get(path) != digest:
            raise SystemExit(f"{path}: {digest} is not the frozen {expected.get(path)}")
    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    pk1 = load_partition(args.pk1_a0s, "train")
    lengths = {row["id"]: len(encode(row, tokenizer, 1 << 30)["ids"]) for row in pk1}
    train = sorted(
        (row for row in pk1 if lengths[row["id"]] <= args.limit), key=lambda r: r["id"]
    )
    control = read_jsonl(args.control)
    base_ids = {row["id"] for row in control if not row["id"].endswith(SUFFIX)}
    if {row["id"] for row in train} != base_ids:
        raise SystemExit("pk1 A0s admission differs from the control's A0s rows")
    resample = sorted(
        row["id"][: -len(SUFFIX)] for row in control if row["id"].endswith(SUFFIX)
    )
    by_id = {row["id"]: row for row in train}
    pool = [by_id[i] for i in resample]
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    files, teachers = {}, {}
    train_path = args.output_dir / "pk1-a0s-4k.train.jsonl"
    files[train_path.name] = write(train_path, train)
    for spec in args.teacher:
        name, path = spec.split("=", 1)
        targets = {row["id"]: row for row in read_jsonl(Path(path))}
        replay, report = attach(pool, targets)
        for row in replay:
            validate_row(row, "train", replay=True)
        out = args.output_dir / f"replay-{name}.jsonl"
        files[out.name] = write(out, replay)
        teachers[name] = {"targets_sha256": hashes[path], **report}
    manifest = {
        "schema": "decision2-27b-m2-replay/1",
        "limit": args.limit,
        "inputs_sha256": hashes,
        "train": {
            "rows": len(train),
            "tokens": sum(lengths[r["id"]] for r in train),
            "renumbered_rows": sum(
                "option_key_renumbering" in r["audit_metadata"] for r in train
            ),
        },
        "replay_pool": {
            "rows": len(pool),
            "tokens": sum(lengths[r["id"]] for r in pool),
            "renumbered_rows": sum(
                "option_key_renumbering" in r["audit_metadata"] for r in pool
            ),
        },
        "trainer_replay_fraction": "1/3 (pool fully used; 10,748 examples, 672 updates)",
        "teachers": teachers,
        "files_sha256": files,
    }
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    (args.output_dir / "REPLAY.json").write_text(text, encoding="utf-8")
    print(text)


def write(path: Path, rows: list[dict[str, Any]]) -> str:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("".join(canonical(row) + "\n" for row in rows))
    return sha_file(path)


if __name__ == "__main__":
    main()
