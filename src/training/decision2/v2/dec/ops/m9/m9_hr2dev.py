"""Decoder M9 HR2 DEV slice diagnostic (prereg dec-m9-prereg-2026-10-01.md, "Development readout path").

  prompts  (launch.sh --cpu) each row of the pinned `hr2.dev.jsonl` becomes one gold-free prompt whose single
           question renders exactly as the row (`ops/m8/m8_data.to_prompt`); writes <out>/hr2-dev.prompts.jsonl, the
           gold file <out>/hr2-dev.gold.jsonl ({id, family, task_type, group_id, language, question, gold}) and
           <out>/manifest.json. The caller moves the gold under /data/dev2/private (mode 600).
  score    (node A host) per family accuracy (invalid answers count as wrong; Score adds the mean absolute level
           error) with `benchmark.score.evaluate_answer`, the family macro and per-type macros; with --reference, the
           paired group bootstrap (2,000 draws, seed 20261001) of the family-macro difference.

A diagnostic only: never a gate or selection criterion; in-distribution for the HR2 arm.

usage: m9_hr2dev.py prompts --dev F --dev-sha S --output DIR
       m9_hr2dev.py score --gold G --predictions P --name N [--reference R --reference-name RN] --output OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))

SCHEMA = "dec-m9-hr2dev/1"
QUESTION = "q"
REPLICATES = 2000
SEED = 20261001


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def gold_value(row: dict[str, Any]) -> Any:
    key = row["options"][row["label"]]["key"]
    if row["task_type"] == "noul":
        if key not in ("true", "false"):
            raise ValueError(f"{row['id']}: Noul option key {key!r}")
        return key == "true"
    if row["task_type"] == "score":
        return int(key)
    return key


def build_prompts(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m8"))
    from m8_data import to_prompt

    prompts, gold = [], []
    for row in rows:
        prompt = to_prompt(row)
        prompts.append(prompt)
        keys = [o["key"] for o in row["options"]]
        gold.append(
            {
                "id": row["id"],
                "family": row["family"],
                "task_type": row["task_type"],
                "group_id": row["group_id"],
                "language": row["language"],
                "question": prompt["questions"][QUESTION],
                "gold": {
                    "value": gold_value(row),
                    "label_to_semantic": {k: k for k in keys},
                },
            }
        )
    return prompts, gold


def outcomes(gold: list[dict], predictions: dict[str, dict]) -> dict[str, dict]:
    from benchmark.score import evaluate_answer

    out = {}
    for item in gold:
        answer = (predictions.get(item["id"]) or {}).get("answers", {}).get(QUESTION)
        result = evaluate_answer(item["question"], item["gold"], answer)
        ok = result.get("status") == "ok"
        out[item["id"]] = {
            "correct": bool(ok and result["correct"]),
            "valid": ok,
            "abs_error": result.get("absolute_error"),
        }
    return out


def summarize(gold: list[dict], res: dict[str, dict]) -> dict[str, Any]:
    fam: dict[str, list[dict]] = defaultdict(list)
    for item in gold:
        fam[item["family"]].append(item)
    families = {}
    for name, items in sorted(fam.items()):
        n = len(items)
        block = {
            "task_type": items[0]["task_type"],
            "n": n,
            "accuracy": sum(res[i["id"]]["correct"] for i in items) / n,
            "valid": sum(res[i["id"]]["valid"] for i in items) / n,
        }
        errs = [
            res[i["id"]]["abs_error"]
            for i in items
            if res[i["id"]]["abs_error"] is not None
        ]
        if items[0]["task_type"] == "score" and errs:
            block["mean_abs_level_error_valid"] = sum(errs) / len(errs)
        families[name] = block
    by_type: dict[str, list[float]] = defaultdict(list)
    for block in families.values():
        by_type[block["task_type"]].append(block["accuracy"])
    return {
        "families": families,
        "family_macro": sum(b["accuracy"] for b in families.values()) / len(families),
        "type_macro": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
        "items": len(gold),
    }


def macro(gold_by_family: dict[str, list[str]], correct: dict[str, int]) -> float:
    return sum(
        sum(correct[i] for i in ids) / len(ids) for ids in gold_by_family.values()
    ) / len(gold_by_family)


def paired(
    gold: list[dict], left: dict[str, dict], right: dict[str, dict]
) -> dict[str, Any]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in gold:
        groups[item["group_id"]].append(item)
    keys = sorted(groups)
    rng = random.Random(SEED)
    diffs = []
    for _ in range(REPLICATES):
        fam_l: dict[str, list[int]] = defaultdict(list)
        fam_r: dict[str, list[int]] = defaultdict(list)
        for g in (keys[rng.randrange(len(keys))] for _ in keys):
            for item in groups[g]:
                fam_l[item["family"]].append(left[item["id"]]["correct"])
                fam_r[item["family"]].append(right[item["id"]]["correct"])
        ml = sum(sum(v) / len(v) for v in fam_l.values()) / len(fam_l)
        mr = sum(sum(v) / len(v) for v in fam_r.values()) / len(fam_r)
        diffs.append(ml - mr)
    diffs.sort()
    return {
        "ci95": [diffs[int(0.025 * REPLICATES)], diffs[int(0.975 * REPLICATES) - 1]],
        "p_le_0": sum(d <= 0 for d in diffs) / REPLICATES,
        "replicates": REPLICATES,
        "seed": SEED,
        "resample": "groups (group_id), family macro recomputed per draw",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("prompts")
    b.add_argument("--dev", type=Path, required=True)
    b.add_argument("--dev-sha", required=True)
    b.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--gold", type=Path, required=True)
    s.add_argument("--predictions", type=Path, required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--reference", type=Path)
    s.add_argument("--reference-name")
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "prompts":
        if sha(a.dev) != a.dev_sha:
            raise ValueError(f"{a.dev}: sha256 differs from {a.dev_sha}")
        rows = read_jsonl(a.dev)
        prompts, gold = build_prompts(rows)
        a.output.mkdir(parents=True, exist_ok=False)
        for name, records in (
            ("hr2-dev.prompts.jsonl", prompts),
            ("hr2-dev.gold.jsonl", gold),
        ):
            with (a.output / name).open("x", encoding="utf-8") as f:
                for r in records:
                    f.write(
                        json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
        manifest = {
            "schema": SCHEMA,
            "dev_sha256": a.dev_sha,
            "rows": len(rows),
            "prompts_sha256": sha(a.output / "hr2-dev.prompts.jsonl"),
            "gold_sha256": sha(a.output / "hr2-dev.gold.jsonl"),
            "families": dict(sorted(_count(g["family"] for g in gold).items())),
        }
        (a.output / "manifest.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n"
        )
        print(json.dumps({k: manifest[k] for k in ("rows", "prompts_sha256")}))
        return 0
    gold = read_jsonl(a.gold)
    preds = {r["id"]: r for r in read_jsonl(a.predictions)}
    res = outcomes(gold, preds)
    out = {
        "schema": SCHEMA,
        "role": "diagnostic (HR2 DEV slice; in-distribution for the HR2 arm); never a gate or selection criterion",
        "name": a.name,
        "files": {"gold": sha(a.gold), "predictions": sha(a.predictions)},
        **summarize(gold, res),
    }
    if a.reference is not None:
        ref_preds = {r["id"]: r for r in read_jsonl(a.reference)}
        ref = outcomes(gold, ref_preds)
        ref_sum = summarize(gold, ref)
        out["reference"] = {
            "name": a.reference_name,
            "file": sha(a.reference),
            "family_macro": ref_sum["family_macro"],
            "type_macro": ref_sum["type_macro"],
            "families": {k: v["accuracy"] for k, v in ref_sum["families"].items()},
        }
        out["delta_family_macro"] = out["family_macro"] - ref_sum["family_macro"]
        out["paired"] = paired(gold, res, ref)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "name": a.name,
                "family_macro": round(out["family_macro"], 4),
                "delta": (
                    None
                    if "delta_family_macro" not in out
                    else round(out["delta_family_macro"], 4)
                ),
            }
        )
    )
    return 0


def _count(values):
    out: dict[str, int] = defaultdict(int)
    for v in values:
        out[v] += 1
    return out


if __name__ == "__main__":
    sys.exit(main())
