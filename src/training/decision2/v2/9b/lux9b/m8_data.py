"""Milestone 8 CPU data builds for the 9B track: DEV2.0-27B (A20r) teacher targets on the K-mix top-up rows.

* ``prompts``: gold-free teacher prompts for the control's TRAIN (M7 arm C, the K-mix top-up rows of
  the five K seeds' continuation), one prompt per row: ``{"id", "state", "questions": {"decision":
  question}}`` with the row's instructions and its options as criteria in row order (Score levels as
  an ordered list). The scored runtime's ``question_to_row`` must rebuild the row's state, type,
  instructions and (key, description) options exactly, which fixes the prompt text (``segments``
  reads nothing else). Rows go to ``shards`` files by position modulo the shard count. Also the
  human-rated rows S of that TRAIN under M6's frozen soft-target rule (``human.jsonl``: id, pool,
  native tokens), which D2 teaches.
* ``teacher``: the D1 / D2 TRAIN and teacher files from the scored runtime's predictions (T = 1):
  D1 gets the A20r distribution over the row's option keys on every row, D2 on the S rows only (the
  other rows train on gold). Noul distributions come from P(true). Each arm's train.jsonl is the
  control's TRAIN byte for byte. The manifest records coverage, the teacher identity bound by the
  prediction manifests, and diagnostics against gold and the control's own-Lux targets (never
  selection).

    python3 -m lux9b.m8_data prompts --spec SPEC --root NAME=PATH ... --output-dir OUT
    python3 -m lux9b.m8_data teacher --spec SPEC --root NAME=PATH ... --prompts-dir P \
        --predictions SHARD=FILE ... --output-dir OUT

Every input is hash-verified; outputs go to a new directory with ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lux9b.m3_data import read_jsonl, verified, write_lines
from lux9b.m6_data import human_rated
from training.model.data import file_sha256, load_partition
from training.model.infer import prompt_input_sha256, question_to_row
from v2.common import eval_only

SCHEMA = "decision2-9b-m8-data/1"
QID = "decision"
RENDER_FIELDS = ("state", "task_type", "instructions")


def prompt_of(row: dict[str, Any]) -> dict[str, Any]:
    """The gold-free runtime prompt that renders exactly as ``row``."""
    kind = row["task_type"]
    keys = [o["key"] for o in row["options"]]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{row['id']}: duplicate option key")
    if kind == "score":
        if keys != [str(i) for i in range(len(keys))]:
            raise ValueError(f"{row['id']}: Score keys are not 0..n-1 in order")
        criteria: Any = [o["description"] for o in row["options"]]
    else:
        criteria = {o["key"]: o["description"] for o in row["options"]}
    question = {"type": kind, "instructions": row["instructions"], "criteria": criteria}
    return {"id": row["id"], "state": row["state"], "questions": {QID: question}}


def render_view(row: dict[str, Any]) -> tuple:
    return (
        *(
            json.dumps(row[f], ensure_ascii=False, sort_keys=False)
            for f in RENDER_FIELDS
        ),
        tuple(
            (o["key"], json.dumps(o["description"], ensure_ascii=False))
            for o in row["options"]
        ),
    )


def check_render(row: dict[str, Any], prompt: dict[str, Any]) -> None:
    """The runtime row rebuilt from ``prompt`` carries the same render inputs as ``row``."""
    rebuilt = question_to_row(prompt, QID, prompt["questions"][QID])
    if render_view(rebuilt) != render_view(row):
        raise ValueError(f"{row['id']}: the runtime prompt renders differently")
    try:
        from training.model.decision_model import segments
    except ImportError:  # torch-free checkout: the field check above is sufficient
        return
    if segments(rebuilt) != segments(row):
        raise ValueError(f"{row['id']}: the runtime prompt text differs")


def prompt_line(row: dict[str, Any]) -> str:
    """The prompt as one JSON line in insertion order (criteria order is option order; the
    runtime digests a prompt in its stored key order), checked after a JSON round trip.
    """
    line = json.dumps(
        prompt_of(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    check_render(row, json.loads(line))
    return line


def write_raw(path: Path, lines: list[str]) -> str:
    with path.open("x", encoding="utf-8") as stream:
        for line in lines:
            stream.write(line + "\n")
    return file_sha256(path)


def load_ids(spec, roots, inputs) -> dict[str, dict[str, Any]]:
    return {e["id"]: e for e in read_jsonl(verified(spec["ids"], roots, inputs))}


def load_control(spec, roots, inputs) -> list[dict[str, Any]]:
    path = verified(spec["control"]["train"], roots, inputs)
    rows = load_partition(path, "train")
    eval_only.check_rows(rows)
    return rows


def human_set(spec, rows, ids) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        entry = ids.get(row["id"])
        if entry is None or entry["source"] != row["source"]:
            raise ValueError(
                f"{row['id']}: not in the XL r2 ids file with the same source"
            )
        if human_rated(row, entry["pool"], spec):
            out.append(
                {"id": row["id"], "pool": entry["pool"], "native": entry["native"]}
            )
    return out


def prompts(spec, roots, out: Path) -> dict[str, Any]:
    inputs: dict[str, str] = {}
    rows = load_control(spec, roots, inputs)
    ids = load_ids(spec, roots, inputs)
    shards = int(spec["shards"])
    lines: list[list[str]] = [[] for _ in range(shards)]
    members: list[list[str]] = [[] for _ in range(shards)]
    for position, row in enumerate(rows):
        lines[position % shards].append(prompt_line(row))
        members[position % shards].append(row["id"])
    human = human_set(spec, rows, ids)
    out.mkdir(parents=True)
    shard_meta = []
    for k, part in enumerate(lines):
        sha = write_raw(out / f"prompts-{k}.jsonl", part)
        shard_meta.append(
            {
                "shard": k,
                "file": f"prompts-{k}.jsonl",
                "rows": len(part),
                "sha256": sha,
                "native_tokens": sum(ids[i]["native"] for i in members[k]),
            }
        )
    human_sha = write_lines(out / "human.jsonl", human)
    human_ids = {h["id"] for h in human}
    by_type = Counter(r["task_type"] for r in rows)
    return {
        "schema": SCHEMA,
        "step": "prompts",
        "name": spec["name"],
        "inputs_sha256": inputs,
        "question_id": QID,
        "rows": len(rows),
        "native_tokens": sum(ids[r["id"]]["native"] for r in rows),
        "rows_by_type": dict(sorted(by_type.items())),
        "render_check": "question_to_row rebuilds state, task_type, instructions and (key, description) options exactly for every row",
        "shards": shard_meta,
        "human": {
            "rule": "M6 soft_target_rule (lux9b.m6_data.human_rated) on the XL r2 pool of each row",
            "rows": len(human),
            "native_tokens": sum(h["native"] for h in human),
            "rows_by_pool": dict(sorted(Counter(h["pool"] for h in human).items())),
            "rows_by_type": dict(
                sorted(
                    Counter(
                        r["task_type"] for r in rows if r["id"] in human_ids
                    ).items()
                )
            ),
            "file": "human.jsonl",
            "sha256": human_sha,
        },
    }


def teacher_probs(row: dict[str, Any], answer: dict[str, Any]) -> dict[str, float]:
    """The teacher distribution over ``row``'s option keys from a scored-runtime answer."""
    if not isinstance(answer, dict) or "error" in answer:
        raise ValueError(f"{row['id']}: invalid teacher answer {answer!r}")
    keys = [o["key"] for o in row["options"]]
    if answer.get("type") != row["task_type"]:
        raise ValueError(f"{row['id']}: teacher answer type differs")
    if row["task_type"] == "noul":
        p = float(answer["noul"])
        probs = {"true": p, "false": 1.0 - p}
    else:
        probs = {k: float(v) for k, v in answer["probabilities"].items()}
    if set(probs) != set(keys):
        raise ValueError(f"{row['id']}: teacher keys differ from the row's options")
    values = [probs[k] for k in keys]
    if (
        any(not math.isfinite(v) or v < 0 for v in values)
        or abs(math.fsum(values) - 1) > 1e-9
    ):
        raise ValueError(f"{row['id']}: invalid teacher distribution")
    return {k: probs[k] for k in keys}


def argmax(probs: dict[str, float], keys: list[str]) -> int:
    values = [probs[k] for k in keys]
    return values.index(max(values))


def entropy(probs: dict[str, float]) -> float:
    return -math.fsum(p * math.log(p) for p in probs.values() if p > 0)


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    return math.fsum(
        pv * (math.log(pv) - math.log(max(q[k], 1e-12)))
        for k, pv in p.items()
        if pv > 0
    )


def check_manifest(manifest: dict[str, Any], want: dict[str, Any]) -> None:
    if manifest.get("model_sha256") != want["model_sha256"]:
        raise ValueError("predictions come from another teacher checkpoint")
    if manifest.get("max_length") != want["max_length"]:
        raise ValueError("predictions were collected at another input limit")
    cal = manifest.get("calibration") or {}
    if cal.get("file_sha256") != want["calibration_sha256"]:
        raise ValueError("predictions carry another calibration file")
    temps = cal.get("temperature_by_type") or {}
    if set(temps) != {"choice", "noul", "score"} or any(
        v != 1.0 for v in temps.values()
    ):
        raise ValueError("teacher predictions are not at T = 1")
    counts = manifest.get("counts") or {}
    if (
        counts.get("invalid_questions", 1) != 0
        or counts.get("truncated_questions", 1) != 0
    ):
        raise ValueError(
            f"teacher predictions have invalid or truncated questions: {counts}"
        )


def teacher(
    spec, roots, prompts_dir: Path, predictions: dict[int, Path], out: Path
) -> dict[str, Any]:
    inputs: dict[str, str] = {}
    rows = load_control(spec, roots, inputs)
    control_teacher = {
        r["id"]: r
        for r in read_jsonl(verified(spec["control"]["teacher"], roots, inputs))
    }
    pmeta = json.loads((prompts_dir / "manifest.json").read_text(encoding="utf-8"))
    train_key = spec["control"]["train"]["file"]
    if pmeta["inputs_sha256"].get(train_key) != inputs[train_key] or pmeta[
        "rows"
    ] != len(rows):
        raise ValueError("the prompts build belongs to another control TRAIN")
    if sorted(predictions) != [s["shard"] for s in pmeta["shards"]]:
        raise ValueError("need exactly one predictions file per prompt shard")
    want = spec["teacher"]
    prompt_by_id: dict[str, dict[str, Any]] = {}
    answer_by_id: dict[str, dict[str, Any]] = {}
    shard_receipts = []
    for s in pmeta["shards"]:
        ppath = prompts_dir / s["file"]
        if file_sha256(ppath) != s["sha256"]:
            raise ValueError(f"prompt shard {s['shard']} changed")
        shard_prompts = read_jsonl(ppath)
        for p in shard_prompts:
            prompt_by_id[p["id"]] = p
        path = predictions[s["shard"]]
        manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text())
        check_manifest(manifest, want)
        if manifest.get("input_sha256") != s["sha256"]:
            raise ValueError(
                f"shard {s['shard']} predictions scored another prompt file"
            )
        preds = read_jsonl(path)
        if [r["id"] for r in preds] != [p["id"] for p in shard_prompts]:
            raise ValueError(f"shard {s['shard']} predictions do not match its prompts")
        for rec, p in zip(preds, shard_prompts):
            if rec["input_sha256"] != prompt_input_sha256(p):
                raise ValueError(
                    f"{p['id']}: prediction input hash differs from its prompt"
                )
            if (
                rec.get("adapter_status") != "ok"
                or rec.get("model_sha256") != want["model_sha256"]
            ):
                raise ValueError(
                    f"{p['id']}: prediction status or teacher identity differs"
                )
            answer_by_id[p["id"]] = rec["answers"][QID]
        shard_receipts.append(
            {
                "shard": s["shard"],
                "predictions_sha256": file_sha256(path),
                "manifest_sha256": file_sha256(
                    path.with_name(path.name + ".manifest.json")
                ),
                "rows": len(preds),
                "counts": manifest.get("counts"),
            }
        )
    if set(answer_by_id) != {r["id"] for r in rows}:
        raise ValueError("teacher predictions do not cover the control TRAIN exactly")
    human_path = prompts_dir / "human.jsonl"
    if file_sha256(human_path) != pmeta["human"]["sha256"]:
        raise ValueError("human.jsonl changed")
    human = {h["id"] for h in read_jsonl(human_path)}
    records: dict[str, dict[str, Any]] = {}
    stats: dict[str, Counter] = defaultdict(Counter)
    sums: dict[str, defaultdict] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        probs = teacher_probs(row, answer_by_id[row["id"]])
        records[row["id"]] = {
            "id": row["id"],
            "input_sha256": row["input_sha256"],
            "teacher_probs": probs,
        }
        keys = [o["key"] for o in row["options"]]
        own = control_teacher[row["id"]]
        if own["input_sha256"] != row["input_sha256"]:
            raise ValueError(f"{row['id']}: control teacher input hash differs")
        own_probs = {k: float(own["teacher_probs"][k]) for k in keys}
        a20r_arg, own_arg = argmax(probs, keys), argmax(own_probs, keys)
        for group in (row["task_type"], "S" if row["id"] in human else "notS", "all"):
            stats[group]["n"] += 1
            stats[group]["a20r_gold"] += a20r_arg == row["label"]
            stats[group]["own_gold"] += own_arg == row["label"]
            stats[group]["a20r_own"] += a20r_arg == own_arg
            sums[group]["a20r_max"] += max(probs.values())
            sums[group]["own_max"] += max(own_probs.values())
            sums[group]["a20r_entropy"] += entropy(probs)
            sums[group]["own_entropy"] += entropy(own_probs)
            sums[group]["kl_a20r_own"] += kl(probs, own_probs)
            if row["task_type"] == "noul":
                gold_true = keys[row["label"]] == "true"
                stats[group]["noul_n"] += 1
                stats[group]["noul_gold_yes"] += gold_true
                stats[group]["noul_a20r_yes"] += probs["true"] > 0.5
                stats[group]["noul_own_yes"] += own_probs["true"] > 0.5
    diagnostics = {}
    for group, c in sorted(stats.items()):
        n = c["n"]
        d = {
            "n": n,
            "accuracy_vs_gold": {
                "a20r": c["a20r_gold"] / n,
                "own_lux": c["own_gold"] / n,
            },
            "argmax_agreement_a20r_own_lux": c["a20r_own"] / n,
            "mean_max_probability": {
                "a20r": sums[group]["a20r_max"] / n,
                "own_lux": sums[group]["own_max"] / n,
            },
            "mean_entropy": {
                "a20r": sums[group]["a20r_entropy"] / n,
                "own_lux": sums[group]["own_entropy"] / n,
            },
            "mean_kl_a20r_to_own_lux": sums[group]["kl_a20r_own"] / n,
        }
        if c["noul_n"]:
            m = c["noul_n"]
            d["noul_yes_rate"] = {
                "gold": c["noul_gold_yes"] / m,
                "a20r": c["noul_a20r_yes"] / m,
                "own_lux": c["noul_own_yes"] / m,
                "n": m,
            }
        diagnostics[group] = d
    out.mkdir(parents=True)
    control_train = verified(spec["control"]["train"], roots, {})
    arms = {}
    for arm, covered in (
        ("D1", [r["id"] for r in rows]),
        ("D2", [r["id"] for r in rows if r["id"] in human]),
    ):
        (out / arm).mkdir()
        shutil.copyfile(control_train, out / arm / "train.jsonl")
        train_sha = file_sha256(out / arm / "train.jsonl")
        if train_sha != spec["control"]["train"]["sha256"]:
            raise ValueError(f"{arm} train.jsonl is not the control's TRAIN")
        teacher_sha = write_lines(
            out / arm / "teacher.jsonl", [records[i] for i in covered]
        )
        covered_set = set(covered)
        arms[arm] = {
            "train_sha256": train_sha,
            "train_rows": len(rows),
            "teacher_sha256": teacher_sha,
            "teacher_rows": len(covered),
            "gold_only_rows": len(rows) - len(covered),
            "teacher_rows_by_type": dict(
                sorted(
                    Counter(
                        r["task_type"] for r in rows if r["id"] in covered_set
                    ).items()
                )
            ),
        }
    return {
        "schema": SCHEMA,
        "step": "teacher",
        "name": spec["name"],
        "inputs_sha256": inputs,
        "prompts_manifest_sha256": file_sha256(prompts_dir / "manifest.json"),
        "teacher": want,
        "shards": shard_receipts,
        "arms": arms,
        "diagnostics": diagnostics,
        "role": "teacher targets; diagnostics are never selection",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=("prompts", "teacher"))
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--root", action="append", required=True, help="name=<path>")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompts-dir", type=Path)
    ap.add_argument("--predictions", action="append", default=[], help="SHARD=<file>")
    args = ap.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} exists")
    roots = {}
    for item in args.root:
        name, _, path = item.partition("=")
        roots[name] = Path(path)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    eval_only.guard(
        args, {k: v for k, v in spec.items() if k not in ("note", "teacher")}
    )
    if args.step == "prompts":
        manifest = prompts(spec, roots, args.output_dir)
    else:
        if not args.prompts_dir or not args.predictions:
            ap.error("teacher needs --prompts-dir and --predictions")
        preds = {}
        for item in args.predictions:
            shard, _, path = item.partition("=")
            preds[int(shard)] = Path(path)
        manifest = teacher(spec, roots, args.prompts_dir, preds, args.output_dir)
    manifest["spec_sha256"] = file_sha256(args.spec)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    keep = ("step", "rows", "native_tokens", "human", "arms")
    print(json.dumps({k: manifest[k] for k in keep if k in manifest}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
