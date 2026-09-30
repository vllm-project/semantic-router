"""Decoder M8 A20r teacher targets (prereg dec-m8-prereg-2026-09-30.md, "Teacher targets"), CPU on node A.

Each label shard's input is the first N typed-final gold-free prompts followed by the shard's slice prompts, run
through A20r's unchanged scored collector (`v2.27b.typed_collect_kernel`, T = 1).

  smoke    the shard's first N predictions must equal A20r's stored formal typed-final predictions: the same answer
           (Choice key, Score argmax, Noul side) and every probability within --tolerance (1e-4)
  convert  the slice predictions of every shard -> teacher records {id, input_sha256, teacher_probs} keyed by the
           TRAIN row's option keys (Choice / Score: the answer's probabilities; Noul: true = noul, false = 1 - noul):
           teacher/D1/teacher.jsonl (every labeled slice row) and teacher/D2/teacher.jsonl (human-rated rows only),
           plus a provenance and coverage manifest (agreement with gold and with own-Lux by type and class)

usage: python3 m8_teacher.py smoke --stored S --shard-predictions P [--items 80] [--tolerance 1e-4] --output O
       python3 m8_teacher.py convert --data-dir D --shard-predictions P0 --shard-predictions P1 ... \
           --smoke R0 --smoke R1 ... --provenance PROV.json --output-dir OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from training.model.data import file_sha256, load_partition  # noqa: E402
from v2.common import eval_only  # noqa: E402

SCHEMA = "dec-m8-a20r-teacher/1"
QUESTION = "q"


def read_jsonl(path: Path) -> list[Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def distribution(answer: dict[str, Any], keys: list[str]) -> dict[str, float]:
    """The scored probability of every option key; raises on an invalid answer."""
    if "error" in answer:
        raise ValueError(f"invalid answer: {answer['error']}")
    if answer["type"] == "noul":
        p = float(answer["noul"])
        probs = {"true": p, "false": 1.0 - p}
    else:
        probs = {k: float(v) for k, v in answer["probabilities"].items()}
    if set(probs) != set(keys):
        raise ValueError(
            f"answer keys {sorted(probs)} differ from options {sorted(keys)}"
        )
    values = [probs[k] for k in keys]
    if (
        any(not math.isfinite(v) or v < 0 for v in values)
        or abs(math.fsum(values) - 1.0) > 1e-6
    ):
        raise ValueError("answer is not a probability distribution")
    return {k: probs[k] for k in keys}


def decision(answer: dict[str, Any]) -> Any:
    if answer["type"] == "noul":
        return answer["noul"] > 0.5
    probs = answer["probabilities"]
    return max(probs, key=probs.__getitem__)


def smoke(
    stored_path: Path, shard_path: Path, items: int, tolerance: float
) -> dict[str, Any]:
    stored = {r["id"]: r for r in read_jsonl(stored_path)}
    head = read_jsonl(shard_path)[:items]
    if len(head) != items:
        raise ValueError(f"{shard_path}: fewer than {items} predictions")
    changed, drift, compared = [], 0.0, 0
    for record in head:
        ref = stored.get(record["id"])
        if ref is None:
            raise ValueError(f"{record['id']}: not a stored typed-final prediction")
        if set(ref["answers"]) != set(record["answers"]):
            changed.append(record["id"])
            continue
        for qid, answer in record["answers"].items():
            other = ref["answers"][qid]
            compared += 1
            if ("error" in answer) != ("error" in other) or answer.get(
                "type"
            ) != other.get("type"):
                changed.append(f"{record['id']}/{qid}")
                continue
            if "error" in answer:
                continue
            if decision(answer) != decision(other):
                changed.append(f"{record['id']}/{qid}")
            if answer["type"] == "noul":
                drift = max(drift, abs(answer["noul"] - other["noul"]))
            else:
                a, b = answer["probabilities"], other["probabilities"]
                if set(a) != set(b):
                    changed.append(f"{record['id']}/{qid}")
                    continue
                drift = max(drift, max(abs(a[k] - b[k]) for k in a))
    status = "PASS" if not changed and drift <= tolerance else "FAIL"
    return {
        "schema": "dec-m8-label-smoke/1",
        "status": status,
        "items": items,
        "questions": compared,
        "changed": changed,
        "max_drift": drift,
        "tolerance": tolerance,
        "stored_sha256": file_sha256(stored_path),
        "shard_predictions_sha256": file_sha256(shard_path),
    }


def convert(
    rows: list[dict[str, Any]],
    classes: dict[str, str],
    predictions: dict[str, dict[str, Any]],
    lux: dict[str, dict[str, float]],
) -> dict[str, Any]:
    d1, d2, failures = [], [], {}
    agree = defaultdict(lambda: [0, 0])
    lux_agree = defaultdict(lambda: [0, 0])
    confidence = defaultdict(float)
    for row in rows:
        keys = [o["key"] for o in row["options"]]
        record = predictions.get(row["id"])
        if record is None:
            failures[row["id"]] = "no prediction"
            continue
        try:
            probs = distribution(record["answers"][QUESTION], keys)
        except (KeyError, ValueError) as exc:
            failures[row["id"]] = str(exc)
            continue
        out = {
            "id": row["id"],
            "input_sha256": row["input_sha256"],
            "teacher_probs": probs,
        }
        d1.append(out)
        cls = classes[row["id"]]
        if cls == "human":
            d2.append(out)
        best = max(keys, key=probs.__getitem__)
        for key in (row["task_type"], f"{cls}/{row['task_type']}"):
            agree[key][0] += int(best == keys[row["label"]])
            agree[key][1] += 1
            confidence[key] += probs[best]
        if row["id"] in lux:
            other = lux[row["id"]]
            for key in (row["task_type"], f"{cls}/{row['task_type']}"):
                lux_agree[key][0] += int(best == max(keys, key=other.__getitem__))
                lux_agree[key][1] += 1
    return {
        "d1": d1,
        "d2": d2,
        "failures": failures,
        "gold_agreement": {
            k: {"agree": a, "n": n, "rate": a / n}
            for k, (a, n) in sorted(agree.items())
        },
        "mean_max_probability": {k: confidence[k] / agree[k][1] for k in sorted(agree)},
        "own_lux_argmax_agreement": {
            k: {"agree": a, "n": n, "rate": a / n}
            for k, (a, n) in sorted(lux_agree.items())
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return file_sha256(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--stored", type=Path, required=True)
    s.add_argument("--shard-predictions", type=Path, required=True)
    s.add_argument("--items", type=int, default=80)
    s.add_argument("--tolerance", type=float, default=1e-4)
    s.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("convert")
    c.add_argument("--data-dir", type=Path, required=True)
    c.add_argument("--shard-predictions", type=Path, action="append", required=True)
    c.add_argument("--smoke", type=Path, action="append", required=True)
    c.add_argument("--provenance", type=Path, required=True)
    c.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "convert":
        eval_only.guard(a)
    if a.cmd == "smoke":
        out = smoke(a.stored, a.shard_predictions, a.items, a.tolerance)
        with open(a.output, "x") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")
        print(
            json.dumps(
                {k: out[k] for k in ("status", "questions", "max_drift")}
                | {"changed": len(out["changed"])}
            )
        )
        return 0 if out["status"] == "PASS" else 1
    smokes = [json.loads(Path(x).read_text()) for x in a.smoke]
    if len(smokes) != len(a.shard_predictions) or any(
        x["status"] != "PASS" for x in smokes
    ):
        raise SystemExit("every shard needs a PASS smoke receipt")
    for receipt, shard in zip(smokes, a.shard_predictions):
        if receipt["shard_predictions_sha256"] != file_sha256(shard):
            raise SystemExit(
                f"{shard}: smoke receipt belongs to another prediction file"
            )
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    manifest = json.loads((a.data_dir / "manifest.json").read_text())
    train = a.data_dir / "slice/train.jsonl"
    if file_sha256(train) != manifest["files_sha256"]["slice/train.jsonl"]:
        raise SystemExit("slice TRAIN differs from its data manifest")
    rows = load_partition(train, "train")
    eval_only.check_rows(rows)
    classes = {
        e["id"]: e["class"] for e in read_jsonl(a.data_dir / "slice/train.ids.jsonl")
    }
    slice_ids = {r["id"] for r in rows}
    predictions: dict[str, dict[str, Any]] = {}
    for shard, receipt in zip(a.shard_predictions, smokes):
        for record in read_jsonl(shard)[receipt["items"] :]:
            if record["id"] not in slice_ids:
                raise SystemExit(f"{shard}: {record['id']} is not a slice row")
            if record["id"] in predictions:
                raise SystemExit(f"{record['id']} predicted twice")
            predictions[record["id"]] = record
    lux = {
        r["id"]: r["teacher_probs"]
        for r in read_jsonl(a.data_dir / "teacher/C/teacher.jsonl")
    }
    out = convert(rows, classes, predictions, lux)
    files = {
        "D1/teacher.jsonl": write_jsonl(a.output_dir / "D1/teacher.jsonl", out["d1"]),
        "D2/teacher.jsonl": write_jsonl(a.output_dir / "D2/teacher.jsonl", out["d2"]),
    }
    human = sum(1 for r in rows if classes[r["id"]] == "human")
    doc = {
        "schema": SCHEMA,
        "prereg": "dec-m8-prereg-2026-09-30.md",
        "provenance": json.loads(a.provenance.read_text()),
        "provenance_sha256": file_sha256(a.provenance),
        "slice_train_sha256": file_sha256(train),
        "shards": [
            {"predictions_sha256": file_sha256(s), "smoke": r}
            for s, r in zip(a.shard_predictions, smokes)
        ],
        "coverage": {
            "slice_rows": len(rows),
            "labeled_rows": len(out["d1"]),
            "human_rows": human,
            "d2_rows": len(out["d2"]),
            "failures": len(out["failures"]),
            "failure_reasons": dict(Counter(out["failures"].values()).most_common(10)),
        },
        "gold_agreement": out["gold_agreement"],
        "mean_max_probability": out["mean_max_probability"],
        "own_lux_argmax_agreement": out["own_lux_argmax_agreement"],
        "files_sha256": files,
    }
    if out["failures"]:
        write_jsonl(
            a.output_dir / "failures.jsonl",
            [{"id": k, "reason": v} for k, v in sorted(out["failures"].items())],
        )
    (a.output_dir / "manifest.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps({"coverage": doc["coverage"], "files": files}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
