"""Decoder M9 data lock check (prereg dec-m9-prereg-2026-10-01.md, "Data", lock checks), CPU (launch.sh --cpu) on
node A after m9-prep.sh. For the two TRAIN files (m9/data/4b/mix/m9-4b-{H9,C9}) it checks:

- the TRAIN hash equals compose.json's, every row id is unique and the quarantine groups are absent;
- matched tokens: |C9 - H9| / H9 <= 0.5%;
- H9's `hr2` rows are exactly the pinned HR2 TRAIN file (same rows, byte-equal canonical JSON), each once;
- the composed teacher covers every row outside the arm's gold-only pools and no `hr2` row has a target;
- the exposure receipt (r2 payload 2194716a) lists 0 groups for the file;
- the C1 registry guard (C1 name keys against every TRAIN source; family-name hits reported only);
- no TRAIN row's canonical state equals a prompt state of the isolation panels (counts per panel and pool; a hit in
  an `hr2` row fails the lock);
- reported only: how many of M7's N7C filler rows (`--n7c-ids`) are in C9's filler.

Writes <out> and prints PASS / FAIL with the reasons. It never writes READY files.

usage: python3 m9_lock.py --mix DIR --hr2 F --hr2-sha S --teacher-root DIR --exposure-root DIR \
           --panel NAME=FILE ... [--n7c-ids F] --out OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "v2" / "9b"))
from lux9b.m3_data import c1_keys, denied_hits  # noqa: E402
from training.model.data import canonical  # noqa: E402

C1_REGISTRY = CODE / "v2/eval/records/sealed-c1-source-registry-2026-09-28.json"
QUARANTINE = CODE / "v2/dec/ops/m7/specs/m7-quarantine-groups.json"
GOLD_POOLS = {"H9": {"base-gold", "hr2"}, "C9": {"base-gold", "fill-gold"}}
MAX_MISMATCH = 0.005


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(state) -> str:
    return hashlib.sha256(canonical(state).encode("utf-8")).hexdigest()


def panel_states(spec: list[str]) -> dict[str, set[str]]:
    out = {}
    for item in spec:
        name, path = item.split("=", 1)
        with open(path, encoding="utf-8") as stream:
            out[name] = {
                digest(json.loads(line)["state"]) for line in stream if line.strip()
            }
    return out


def check_arm(
    arm: str,
    mix: Path,
    comp: dict,
    hr2_rows: dict[str, str],
    panels: dict[str, set[str]],
    teacher_root: Path,
    exposure_root: Path,
    keys,
    quarantine: set[str],
) -> tuple[dict, list[str]]:
    name = f"m9-4b-{arm}"
    train = mix / name / "train.jsonl"
    fails: list[str] = []
    got = sha(train)
    if got != comp["files"][arm]["train_sha256"]:
        fails.append(f"{name}: TRAIN hash differs from compose.json")
    pools = {}
    with (mix / name / "train.ids.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            pools[entry["id"]] = entry["pool"]
    ids, sources, families = Counter(), Counter(), Counter()
    qhits, rows, hr2_seen, hr2_differ = 0, 0, 0, 0
    panel_hits: dict[str, Counter] = defaultdict(Counter)
    with train.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows += 1
            ids[row["id"]] += 1
            sources[row["source"]] += 1
            families[row["family"]] += 1
            qhits += row["group_id"] in quarantine
            pool = pools.get(row["id"], "?")
            if pool == "hr2":
                hr2_seen += 1
                hr2_differ += hr2_rows.get(row["id"]) != canonical(row)
            d = digest(row["state"])
            for panel, states in panels.items():
                if d in states:
                    panel_hits[panel][pool] += 1
    if any(n > 1 for n in ids.values()):
        fails.append(f"{name}: repeated ids")
    if set(ids) != set(pools):
        fails.append(f"{name}: train.ids.jsonl does not list exactly the TRAIN ids")
    if qhits:
        fails.append(f"{name}: {qhits} quarantined rows")
    if arm == "H9":
        if hr2_seen != len(hr2_rows) or hr2_differ:
            fails.append(
                f"{name}: HR2 block is not the pinned HR2 file ({hr2_seen} rows, {hr2_differ} differ, "
                f"file {len(hr2_rows)})"
            )
    elif hr2_seen:
        fails.append(f"{name}: {hr2_seen} HR2 rows in the control")
    hr2_panel = {p: c["hr2"] for p, c in panel_hits.items() if c.get("hr2")}
    if hr2_panel:
        fails.append(
            f"{name}: HR2 rows share a state with an isolation panel {hr2_panel}"
        )
    exp_path = exposure_root / name / f"exposure-{name}.json"
    exposure = json.loads(exp_path.read_text()) if exp_path.is_file() else None
    if exposure is None:
        fails.append(f"{name}: no exposure receipt")
    elif exposure.get("groups") or exposure["files"][0]["sha256"] != got:
        fails.append(f"{name}: exposure receipt lists groups or another file")
    tman_path = teacher_root / name / "teacher.jsonl.manifest.json"
    tman = json.loads(tman_path.read_text()) if tman_path.is_file() else None
    if tman is None:
        fails.append(f"{name}: no composed teacher")
    else:
        if tman["train_sha256"] != got:
            fails.append(f"{name}: teacher composed for another TRAIN file")
        missing = tman["missing_by_pool"]
        stray = set(missing) - GOLD_POOLS[arm]
        if stray or tman["covered"] + sum(missing.values()) != tman["rows"]:
            fails.append(
                f"{name}: teacher coverage outside the gold-only pools ({sorted(stray)})"
            )
        if arm == "H9" and missing.get("hr2", 0) != len(hr2_rows):
            fails.append(
                f"{name}: {len(hr2_rows) - missing.get('hr2', 0)} HR2 rows have a target"
            )
    c1 = {s: h for s in sources if (h := denied_hits(s, keys, []))}
    c1_family = {f: h for f in families if (h := denied_hits(f, keys, []))}
    if c1:
        fails.append(f"{name}: C1 registry source hits {sorted(c1)}")
    return {
        "train": str(train),
        "train_sha256": got,
        "rows": rows,
        "tokens": comp["report"]["tokens"][arm],
        "pools": comp["files"][arm]["pools"],
        "hr2_rows": hr2_seen,
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
        "panel_state_hits": {p: dict(c) for p, c in sorted(panel_hits.items())},
        "c1_registry_source_hits": c1,
        "c1_registry_family_name_hits": c1_family,
    }, fails


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mix", type=Path, required=True)
    p.add_argument("--hr2", type=Path, required=True)
    p.add_argument("--hr2-sha", required=True)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--exposure-root", type=Path, required=True)
    p.add_argument("--panel", action="append", default=[], metavar="NAME=FILE")
    p.add_argument("--n7c-ids", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    comp = json.loads((a.mix / "compose.json").read_text())
    fails: list[str] = []
    if sha(a.hr2) != a.hr2_sha or comp["inputs_sha256"]["hr2"] != a.hr2_sha:
        fails.append("HR2 file differs from its pinned SHA-256 or from compose.json's")
    hr2_rows = {}
    with a.hr2.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                hr2_rows[row["id"]] = canonical(row)
    quarantine = set(json.loads(QUARANTINE.read_text())["group_ids"])
    keys = c1_keys(json.loads(C1_REGISTRY.read_text()))
    panels = panel_states(a.panel)
    arms = {}
    for arm in ("H9", "C9"):
        arms[arm], arm_fails = check_arm(
            arm,
            a.mix,
            comp,
            hr2_rows,
            panels,
            a.teacher_root,
            a.exposure_root,
            keys,
            quarantine,
        )
        fails += arm_fails
    match = comp["report"]["match"]
    if match["max_relative"] > MAX_MISMATCH:
        fails.append(f"token mismatch {match['max_relative']:.4%} > {MAX_MISMATCH:.1%}")
    n7c = None
    if a.n7c_ids is not None:
        fill_n7c = set()
        with a.n7c_ids.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if entry["pool"].startswith("fill"):
                    fill_n7c.add(entry["id"])
        fill_c9 = set()
        with (a.mix / "m9-4b-C9" / "train.ids.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if entry["pool"].startswith("fill"):
                    fill_c9.add(entry["id"])
        n7c = {
            "n7c_ids_sha256": sha(a.n7c_ids),
            "n7c_fill_rows": len(fill_n7c),
            "in_c9_fill": len(fill_n7c & fill_c9),
            "c9_fill_rows": len(fill_c9),
            "contains_n7c_filler": fill_n7c <= fill_c9,
        }
    doc = {
        "schema": "dec-m9-lock/1",
        "status": "FAIL" if fails else "PASS",
        "fails": fails,
        "compose_sha256": sha(a.mix / "compose.json"),
        "inputs_sha256": comp["inputs_sha256"],
        "match": match,
        "added_tokens_H9": comp["report"]["added_tokens_H9"],
        "blocks": {
            k: comp["report"][k] for k in ("base", "hr2", "quarantine_rows_dropped")
        },
        "filler": {
            k: v for k, v in comp["report"]["filler"].items() if k != "group_ids"
        },
        "n7c_filler": n7c,
        "panels": {
            item.split("=", 1)[0]: {
                "file": item.split("=", 1)[1],
                "sha256": sha(Path(item.split("=", 1)[1])),
                "states": len(panels[item.split("=", 1)[0]]),
            }
            for item in a.panel
        },
        "arms": arms,
        "c1_registry_sha256": sha(C1_REGISTRY),
        "c1_keys": len(keys),
    }
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": doc["status"],
                "fails": fails,
                "tokens": comp["report"]["tokens"],
                "n7c_filler": n7c,
            }
        )
    )
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
