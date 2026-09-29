"""Decoder M7 data lock check (prereg dec-m7-prereg-2026-09-30.md, "Data lock"), host Python on node B after
m7-prep.sh (CPU, stdlib only). For each tier's three TRAIN files (m7/data/<tier>/mix/m7-<tier>-<ARM>) it checks:

- the TRAIN hash equals compose.json's, every row id is unique, and the quarantine groups are absent;
- no HS1 row of a defect group (the `--hs1-drop-substring` of compose.json) is present;
- matched tokens: |C - H| / H and |P - H| / H <= 0.5%, and P's filler is nested in C's (compose.json);
- the exposure receipt (r2 payload 2194716a) lists 0 groups for the file;
- the composed teacher covers every row except the gold-only pools, with the argmax agreement reported;
- the C1 registry guard (the 9B rule: C1 name keys against every TRAIN source; family-name hits reported only).

Writes <out> (lock-<tier>.json) and prints PASS / FAIL with the reasons. It never writes READY files.

usage: python3 m7_lock.py --tier 4b|2b --out /data/dev2/runs/dec/m7/lock-<tier>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE / "v2" / "9b"))
from lux9b.m3_data import c1_keys, denied_hits  # noqa: E402

M = Path("/data/dev2/runs/dec/m7")
C1_REGISTRY = CODE / "v2/eval/records/sealed-c1-source-registry-2026-09-28.json"
QUARANTINE = CODE / "v2/dec/ops/m7/specs/m7-quarantine-groups.json"
GOLD_POOLS = {
    "4b": {"base-gold", "lp-gold", "fill-gold", "hs1", "pn1"},
    "2b": {"hs1", "pn1"},
}
MAX_MISMATCH = 0.005


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tier", choices=sorted(GOLD_POOLS), required=True)
    p.add_argument("--root", type=Path, default=M)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    mix = a.root / "data" / a.tier / "mix"
    comp = json.loads((mix / "compose.json").read_text())
    report = comp["report"]
    quarantine = set(json.loads(QUARANTINE.read_text())["group_ids"])
    drops = comp["args"].get("hs1_drop_substring") or []
    keys = c1_keys(json.loads(C1_REGISTRY.read_text()))
    fails: list[str] = []
    arms = {}
    for arm in ("H", "C", "P"):
        name = f"m7-{a.tier}-{arm}"
        train = mix / name / "train.jsonl"
        got = sha(train)
        if got != comp["files"][arm]["train_sha256"]:
            fails.append(f"{name}: TRAIN hash differs from compose.json")
        ids, sources, families, qhits, defect = Counter(), Counter(), Counter(), 0, 0
        tokens_rows = 0
        with train.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                ids[row["id"]] += 1
                sources[row["source"]] += 1
                families[row["family"]] += 1
                qhits += row["group_id"] in quarantine
                defect += row["source"] == "decision2_hardskills_hs1" and any(
                    s in line for s in drops
                )
                tokens_rows += 1
        if any(n > 1 for n in ids.values()):
            fails.append(f"{name}: repeated ids")
        if qhits:
            fails.append(f"{name}: {qhits} quarantined rows")
        if defect:
            fails.append(f"{name}: {defect} HS1 defect rows")
        exp_path = a.root / "exposure" / name / f"exposure-{name}.json"
        exposure = json.loads(exp_path.read_text()) if exp_path.is_file() else None
        if exposure is None:
            fails.append(f"{name}: no exposure receipt")
        elif exposure.get("groups") or exposure["files"][0]["sha256"] != got:
            fails.append(f"{name}: exposure receipt lists groups or another file")
        tman_path = a.root / "teacher" / name / "teacher.jsonl.manifest.json"
        tman = json.loads(tman_path.read_text()) if tman_path.is_file() else None
        if tman is None:
            fails.append(f"{name}: no composed teacher")
        else:
            if tman["train_sha256"] != got:
                fails.append(f"{name}: teacher composed for another TRAIN file")
            stray = set(tman["missing_by_pool"]) - GOLD_POOLS[a.tier]
            if (
                stray
                or tman["covered"] + sum(tman["missing_by_pool"].values())
                != tman["rows"]
            ):
                fails.append(
                    f"{name}: teacher coverage outside the gold-only pools ({sorted(stray)})"
                )
        c1 = {s: h for s in sources if (h := denied_hits(s, keys, []))}
        c1_family = {f: h for f in families if (h := denied_hits(f, keys, []))}
        if c1:
            fails.append(f"{name}: C1 registry source hits {sorted(c1)}")
        arms[arm] = {
            "train": str(train),
            "train_sha256": got,
            "rows": tokens_rows,
            "tokens": report["tokens"][arm],
            "pools": comp["files"][arm]["pools"],
            "exposure_sha256": sha(exp_path) if exposure is not None else None,
            "exposure_groups": (
                None if exposure is None else len(exposure.get("groups") or [])
            ),
            "teacher": (
                None
                if tman is None
                else {
                    "sha256": tman["output_sha256"],
                    "covered": tman["covered"],
                    "rows": tman["rows"],
                    "missing_by_pool": tman["missing_by_pool"],
                    "train_label_agreement": tman.get("train_label_agreement"),
                }
            ),
            "c1_registry_source_hits": c1,
            "c1_registry_family_name_hits": c1_family,
        }
    match = report["match"]
    if match["max_relative"] > MAX_MISMATCH:
        fails.append(f"token mismatch {match['max_relative']:.4%} > {MAX_MISMATCH:.1%}")
    if not report["filler"]["P_nested_in_C"]:
        fails.append("P filler not nested in C")
    doc = {
        "schema": "dec-m7-lock/1",
        "tier": a.tier,
        "status": "FAIL" if fails else "PASS",
        "fails": fails,
        "compose_sha256": sha(mix / "compose.json"),
        "inputs_sha256": comp["inputs_sha256"],
        "match": match,
        "added_tokens_H": report["added_tokens_H"],
        "blocks": {
            k: report[k]
            for k in ("base", "hs1", "lp", "pn1", "filler", "quarantine_rows_dropped")
        },
        "arms": arms,
        "c1_registry_sha256": sha(C1_REGISTRY),
        "c1_keys": len(keys),
    }
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "tier": a.tier,
                "status": doc["status"],
                "fails": fails,
                "tokens": report["tokens"],
            }
        )
    )
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
