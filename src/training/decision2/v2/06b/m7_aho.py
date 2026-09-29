"""M7 (a) FIT_AHO / CHK held-out Score rows and the CAL698 Score ids (stdlib only).

    python3 -m v2.06b.m7_aho build --output DIR [--arm NAME=PATH ...] [--mixture PATH ...]
        [--select PATH] [--cal700 PATH] [--cal698 PATH] [--excluded-groups PATH]
        [--panel-root DIR] [--pi-manifest PATH]

M7 prereg section 1.2. AHO = the held-out slices (split `select`) of A7q, H6, G6, A7g, A6g
and A6h; only their Score rows with L in {3, 4, 5}. A row is dropped if its id, input
sha256 or group id is in one of the six M6 training mixtures, SELECT, CAL700 or CAL698; if
its group is one of the r2 rescreen's excluded groups (matched by group key, row id or input
sha256 of the exclusion list); or if its state exactly matches a state of a formal or
development panel (typed FINAL / DEV, CSS15 / CSS pilot, public 231, mlx-diag, HT-DEV; text
states compare as text, structured states as canonical JSON). The rest splits by group:
FIT if int(sha256("m7a-aho-split:" + group)[:16], 16) is even, else CHK.

DIR gets `fit_aho.jsonl` and `chk.jsonl` (the input lines byte for byte), `index.jsonl`
(id, arm, split, levels, language; no gold), `cal698.score.ids.json` (the CAL698 Score row
ids, checked against CAL700 gold) and `MANIFEST.json` (every input path + sha256, drop
counts by reason, counts by arm x L x split x gold level and by arm x split x language, and
an informational PI-v4 projection cross-check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import canonical, file_sha256, write_json

SPLIT_SALT = "m7a-aho-split:"
LEVEL_COUNTS = (3, 4, 5)
CAL698_SHA256 = "19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f"
CAL698_SCORE_ROWS = 90
ARMS = {
    "A7q": "/data/dev2/runs/data/m3b/hf/v2/a7/arms/A7q/aho.jsonl",
    "H6": "/data/dev2/private/data/hf-upload-m2a/m2/arms/H6/aho.jsonl",
    "G6": "/data/dev2/private/data/hf-upload-m2a/m2/arms/G6/aho.jsonl",
    "A7g": "/data/dev2/runs/data/m3b/hf/v2/a7/arms/A7g/aho.jsonl",
    "A6g": "/data/dev2/private/data/arms-v1/wave1-q/A6g.aho.jsonl",
    "A6h": "/data/dev2/private/data/arms-v1/wave2-q/A6h.aho.jsonl",
}
PI_ROLES = {
    "A7q": "a7_aho_A7q",
    "H6": "v2_aho_H6",
    "G6": "v2_aho_G6",
    "A7g": "a7_aho_A7g",
    "A6g": "v1_aho_a6g",
    "A6h": "v1_aho_a6h",
}
MIXTURES = [
    f"/data/dev2/runs/06b/m6/data/m6-{family}-s{seed}.train.jsonl"
    for family in ("mx", "cx")
    for seed in (1, 2, 3)
]
SELECT = "/data/dev2/private/panels/gold/select.jsonl"
CAL700 = "/data/dev2/private/panels/gold/cal.jsonl"
CAL698 = (
    "/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/"
    "snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl"
)
EXCLUDED_GROUPS = "/data/dev2/runs/eval/m5/overlap-effects/final/excluded-groups.json"
PANEL_ROOT = "/data/dev2/private/panels"
PANELS = (
    "typed-final",
    "typed-dev",
    "css15",
    "css-pilot",
    "public231",
    "mlx-diag",
    "ht-dev",
)
PI_MANIFEST = "/data/dev2/runs/data/m3b/gap/c1/pi/manifest.json"
DROP_REASONS = (
    "in_m6_training_mixture",
    "in_select",
    "in_cal",
    "r2_excluded_group",
    "panel_state_match",
)


def split_of(group: str) -> str:
    value = int(hashlib.sha256((SPLIT_SALT + group).encode()).hexdigest()[:16], 16)
    return "fit" if value % 2 == 0 else "chk"


def group_of(row: dict[str, Any]) -> str:
    group = row.get("group_id")
    return group if isinstance(group, str) and group else row["id"]


def state_key(state: Any) -> str:
    return state if isinstance(state, str) else "json:" + canonical(state)


def lines(path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    out = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            text = line.rstrip("\n")
            if not text.strip():
                raise ValueError(f"{path}:{number}: blank JSONL line")
            out.append((text, json.loads(text)))
    return out


class Keys:
    """id / input_sha256 / group_id sets of one reference collection."""

    def __init__(self) -> None:
        self.ids: set[str] = set()
        self.inputs: set[str] = set()
        self.groups: set[str] = set()

    def add(self, row: dict[str, Any]) -> None:
        self.ids.add(row["id"])
        if "input_sha256" in row:
            self.inputs.add(row["input_sha256"])
        if row.get("group_id"):
            self.groups.add(row["group_id"])

    def hit(self, row: dict[str, Any]) -> bool:
        return (
            row["id"] in self.ids
            or row.get("input_sha256") in self.inputs
            or group_of(row) in self.groups
        )


def reference_keys(paths: list[str]) -> Keys:
    keys = Keys()
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    keys.add(json.loads(line))
    return keys


def excluded_keys(path: str) -> Keys:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = Keys()
    for group, entry in data["groups"].items():
        keys.groups.add(group)
        keys.ids.update(entry.get("row_ids", []))
        values = entry.get("input_sha256", [])
        keys.inputs.update([values] if isinstance(values, str) else values)
    return keys


def panel_states(
    root: str, panels: tuple[str, ...]
) -> tuple[dict[str, set[str]], dict]:
    states, inputs = {}, {}
    for panel in panels:
        path = Path(root) / "goldfree" / f"{panel}.prompts.jsonl"
        inputs[panel] = {"path": str(path), "sha256": file_sha256(path)}
        states[panel] = {state_key(row["state"]) for _, row in lines(path)}
    return states, inputs


def pi_crosscheck(
    manifest_path: str, arms: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Informational: the PI-v4 AHO projection has the same ids (and, per row, inputs)."""
    manifest = {e["role"]: e for e in json.loads(Path(manifest_path).read_text())}
    out: dict[str, Any] = {
        "manifest": {"path": manifest_path, "sha256": file_sha256(manifest_path)}
    }
    for arm, rows in arms.items():
        entry = manifest.get(PI_ROLES.get(arm, ""))
        if entry is None:
            out[arm] = {"role": None}
            continue
        path = Path(entry["path"])
        actual = file_sha256(path) if path.is_file() else None
        projected = {r["id"]: r for _, r in lines(path)} if actual else {}
        by_id = {r["id"]: r for r in rows}
        out[arm] = {
            "role": entry["role"],
            "sha256_matches_manifest": actual == entry["sha256"],
            "ids_equal": set(projected) == set(by_id),
            "projection_mismatches": sum(
                1
                for key, r in projected.items()
                if key in by_id
                and any(r.get(f) != by_id[key].get(f) for f in r if f != "id")
            ),
        }
    return out


def nested() -> defaultdict:
    return defaultdict(nested)


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): plain(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))
        }
    return value


def cal698_score_ids(cal698: str, cal700: str) -> tuple[list[str], dict[str, Any]]:
    gold = {row["id"]: row for _, row in lines(cal700)}
    ids, levels = [], Counter()
    for _, row in lines(cal698):
        if row["task_type"] != "score":
            continue
        reference = gold.get(row["id"])
        if (
            reference is None
            or reference["input_sha256"] != row["input_sha256"]
            or reference["label"] != row["label"]
        ):
            raise ValueError(f"{row['id']}: CAL698 Score row differs from CAL700 gold")
        ids.append(row["id"])
        levels[len(row["options"])] += 1
    if len(ids) != CAL698_SCORE_ROWS:
        raise ValueError(
            f"CAL698 has {len(ids)} Score rows, expected {CAL698_SCORE_ROWS}"
        )
    return ids, {"score_rows": len(ids), "levels": dict(levels)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    output: Path = args.output
    if output.exists():
        raise FileExistsError(output)
    arms = dict(entry.split("=", 1) for entry in args.arm) if args.arm else dict(ARMS)
    mixtures = args.mixture or MIXTURES
    if file_sha256(args.cal698) != args.cal698_sha256:
        raise ValueError("CAL698 bytes differ from the preregistered sha256")
    cal_ids, cal_info = cal698_score_ids(args.cal698, args.cal700)

    references = {
        "in_m6_training_mixture": reference_keys(mixtures),
        "in_select": reference_keys([args.select]),
        "in_cal": reference_keys([args.cal700, args.cal698]),
        "r2_excluded_group": excluded_keys(args.excluded_groups),
    }
    states, panel_inputs = panel_states(args.panel_root, tuple(args.panel))

    kept: dict[str, list[str]] = {"fit": [], "chk": []}
    index = []
    first_reason: Counter = Counter()
    any_reason: Counter = Counter()
    panel_hits: Counter = Counter()
    filtered: Counter = Counter()
    counts = nested()
    languages = nested()
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    group_split: dict[str, str] = {}
    for arm, path in arms.items():
        rows = lines(path)
        arm_rows[arm] = [row for _, row in rows]
        for text, row in rows:
            if row["id"] in seen_ids:
                raise ValueError(f"{row['id']}: duplicate id across AHO slices")
            seen_ids.add(row["id"])
            if row.get("split") != "select":
                raise ValueError(f"{row['id']}: AHO rows must be split select")
            levels = len(row["options"])
            if row["task_type"] != "score" or levels not in LEVEL_COUNTS:
                filtered[arm] += 1
                continue
            reasons = [name for name, keys in references.items() if keys.hit(row)]
            hit_panels = [p for p, s in states.items() if state_key(row["state"]) in s]
            if hit_panels:
                reasons.append("panel_state_match")
                panel_hits.update(hit_panels)
            if reasons:
                first_reason[min(reasons, key=DROP_REASONS.index)] += 1
                any_reason.update(reasons)
                continue
            group = group_of(row)
            split = split_of(group)
            if group_split.setdefault(group, split) != split:
                raise AssertionError("a group straddles the split")
            kept[split].append(text)
            index.append(
                {
                    "id": row["id"],
                    "arm": arm,
                    "split": split,
                    "levels": levels,
                    "language": row.get("language"),
                }
            )
            cell = counts[arm][levels][split]
            cell[row["label"]] = cell.get(row["label"], 0) + 1
            lang = languages[arm][split]
            lang[row.get("language")] = lang.get(row.get("language"), 0) + 1

    output.mkdir(parents=True)
    files = {}
    for split, name in (("fit", "fit_aho.jsonl"), ("chk", "chk.jsonl")):
        path = output / name
        with path.open("x", encoding="utf-8") as stream:
            for text in kept[split]:
                stream.write(text + "\n")
        files[name] = {"sha256": file_sha256(path), "rows": len(kept[split])}
    with (output / "index.jsonl").open("x", encoding="utf-8") as stream:
        for entry in index:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    files["index.jsonl"] = {
        "sha256": file_sha256(output / "index.jsonl"),
        "rows": len(index),
    }
    files["cal698.score.ids.json"] = {
        "sha256": write_json(
            output / "cal698.score.ids.json",
            {
                "cal698_sha256": args.cal698_sha256,
                "cal700_sha256": file_sha256(args.cal700),
                "ids": cal_ids,
                **cal_info,
            },
            exclusive=True,
        ),
        "rows": len(cal_ids),
    }
    by_level_split: Counter = Counter()
    for entry in index:
        by_level_split[f"L{entry['levels']}/{entry['split']}"] += 1
    manifest = {
        "schema": "dev2-06b-m7a-aho/1",
        "rule": {
            "split": f'int(sha256("{SPLIT_SALT}" + group)[:16], 16) even -> fit, else chk',
            "group": "group_id, or id if it has none",
            "levels": list(LEVEL_COUNTS),
            "drop_reasons": list(DROP_REASONS),
        },
        "inputs": {
            "arms": {a: {"path": p, "sha256": file_sha256(p)} for a, p in arms.items()},
            "mixtures": [{"path": p, "sha256": file_sha256(p)} for p in mixtures],
            "select": {"path": args.select, "sha256": file_sha256(args.select)},
            "cal700": {"path": args.cal700, "sha256": file_sha256(args.cal700)},
            "cal698": {"path": args.cal698, "sha256": args.cal698_sha256},
            "excluded_groups": {
                "path": args.excluded_groups,
                "sha256": file_sha256(args.excluded_groups),
                "groups": len(references["r2_excluded_group"].groups),
            },
            "panels": panel_inputs,
        },
        "outputs": files,
        "filtered_not_score_L345": dict(filtered),
        "dropped": {
            "total": sum(first_reason.values()),
            "by_first_reason": dict(first_reason),
            "by_any_reason": dict(any_reason),
            "panel_state_matches": dict(panel_hits),
        },
        "kept": {"fit": len(kept["fit"]), "chk": len(kept["chk"]), **by_level_split},
        "counts_arm_levels_split_gold": plain(counts),
        "counts_arm_split_language": plain(languages),
        "groups": {
            "fit": sum(1 for s in group_split.values() if s == "fit"),
            "chk": sum(1 for s in group_split.values() if s == "chk"),
        },
    }
    if args.pi_manifest:
        manifest["pi_v4_crosscheck"] = pi_crosscheck(args.pi_manifest, arm_rows)
    manifest["manifest_of_outputs_sha256"] = hashlib.sha256(
        canonical(files).encode()
    ).hexdigest()
    write_json(output / "MANIFEST.json", manifest, exclusive=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("build")
    one.add_argument("--output", type=Path, required=True)
    one.add_argument("--arm", action="append", default=[], help="NAME=PATH")
    one.add_argument("--mixture", action="append", default=[])
    one.add_argument("--select", default=SELECT)
    one.add_argument("--cal700", default=CAL700)
    one.add_argument("--cal698", default=CAL698)
    one.add_argument("--cal698-sha256", default=CAL698_SHA256)
    one.add_argument("--excluded-groups", default=EXCLUDED_GROUPS)
    one.add_argument("--panel-root", default=PANEL_ROOT)
    one.add_argument("--panel", action="append", default=None)
    one.add_argument("--pi-manifest", default=PI_MANIFEST)
    args = parser.parse_args(argv)
    args.panel = args.panel or list(PANELS)
    manifest = build(args)
    print(
        json.dumps(
            {
                "kept": manifest["kept"],
                "dropped": manifest["dropped"]["by_first_reason"],
                "outputs": {k: v["rows"] for k, v in manifest["outputs"].items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
