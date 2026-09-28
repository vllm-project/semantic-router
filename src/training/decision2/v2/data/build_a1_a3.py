"""Build arm A1 (ABCD, SGD, QASC) or A3 (MuSiQue-Full) from extracted sources.

Projections follow amendment 1 of the data-arms v1 preregistration. Choice
options are rotated with seed "<arm>-v1:<row id>" (amendment 1, section 2)
except musique_final_support, whose options mirror the paragraph order of the
state; a gold position above twice its uniform share is reported as a warning.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import training.model.data as contract
from training.model.data import file_sha256

from v2.data.sources import abcd, common, musique, qasc, sgd
from v2.data.sources.common import cap_groups, write_arm

SCHEMA = "decision2.v2.build_a1_a3.v1"
PREREGISTRATION = (
    "v2/data/records/data-arms-v1-prereg-2026-09-28.md",
    "v2/data/records/data-arms-v1-prereg-amendment-1-2026-09-28.md",
)
ARMS: dict[str, tuple[ModuleType, ...]] = {"a1": (abcd, sgd, qasc), "a3": (musique,)}
SOURCE_DIRS = ("abcd", "sgd", "qasc", "musique")
UNROTATED = frozenset({musique.CHOICE_FAMILY})
CAP_FLAGS = {
    "abcd_choice_cap": abcd.CHOICE_FAMILY,
    "abcd_noul_cap": abcd.NOUL_FAMILY,
    "sgd_choice_cap": sgd.CHOICE_FAMILY,
    "sgd_noul_cap": sgd.NOUL_FAMILY,
    "qasc_cap": qasc.FAMILY,
    "musique_pairs": "musique_pairs",
}


def source_dir(module: ModuleType) -> str:
    return module.__name__.rsplit(".", 1)[-1]


def default_caps(arm: str) -> dict[str, int]:
    return {key: cap for module in ARMS[arm] for key, cap in module.CAPS.items()}


def outputs(out_dir: Path, arm: str) -> list[Path]:
    return [
        out_dir / f"{arm}.{part}" for part in ("train.jsonl", "aho.jsonl", "build.json")
    ]


def code_sha256(modules: Sequence[ModuleType]) -> dict[str, str]:
    paths = {
        "v2.data.build_a1_a3": Path(__file__),
        common.__name__: Path(common.__file__),
        contract.__name__: Path(contract.__file__),
        **{module.__name__: Path(module.__file__) for module in modules},
    }
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def family_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = collections.Counter(row["label"] for row in rows)
    summary: dict[str, Any] = {
        "rows": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "labels": dict(sorted(labels.items())),
    }
    if rows and rows[0]["task_type"] == "noul":
        summary["true_share"] = round(labels[1] / len(rows), 6)
    else:
        by_count: dict[int, collections.Counter[int]] = collections.defaultdict(
            collections.Counter
        )
        for row in rows:
            by_count[len(row["options"])][row["label"]] += 1
        summary["gold_position_by_option_count"] = {
            count: dict(sorted(positions.items()))
            for count, positions in sorted(by_count.items())
        }
    return summary


def position_warnings(family: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    by_count: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    for row in rows:
        by_count[len(row["options"])][row["label"]] += 1
    warnings = []
    for count, positions in sorted(by_count.items()):
        total = sum(positions.values())
        for position, hits in sorted(positions.items()):
            if hits * count > 2 * total:
                warnings.append(
                    f"{family}: gold position {position} holds {hits} of {total} rows "
                    f"with {count} options ({hits / total:.1%}; uniform {1 / count:.1%})"
                )
    return warnings


def build_arm(
    arm: str,
    roots: dict[str, Path],
    out_dir: Path,
    caps: dict[str, int] | None = None,
) -> dict[str, Any]:
    existing = [str(path) for path in outputs(out_dir, arm) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {', '.join(existing)}")
    caps = {**default_caps(arm), **(caps or {})}
    build: dict[str, Any] = {
        "schema": SCHEMA,
        "preregistration": list(PREREGISTRATION),
        "caps": caps,
        "seeds": {},
        "inputs": {},
        "code_sha256": code_sha256(ARMS[arm]),
        "sources": {},
        "families": {},
        "shortfalls": {},
        "warnings": [],
    }
    rows: list[dict[str, Any]] = []
    for module in ARMS[arm]:
        built, report = module.build(roots[source_dir(module)])
        if module is musique:
            kept = musique.select(built, caps["musique_pairs"])
            wanted = {musique.NOUL_FAMILY: 2 * caps["musique_pairs"]}
        else:
            kept = {
                family: cap_groups(members, caps[family], module.SEED)
                for family, members in built.items()
            }
            wanted = {family: caps[family] for family in built}
        build["seeds"][module.SOURCE] = module.SEED
        build["inputs"][module.SOURCE] = report.pop("inputs")
        build["sources"][module.SOURCE] = {
            "candidates": {family: len(members) for family, members in built.items()},
            "kept": {family: len(members) for family, members in kept.items()},
            **report,
        }
        for family, members in kept.items():
            build["families"][family] = family_summary(members)
            if family in wanted and len(members) < wanted[family]:
                build["shortfalls"][family] = {
                    "cap": wanted[family],
                    "kept": len(members),
                }
            if family in UNROTATED:
                build["warnings"] += position_warnings(family, members)
            rows.extend(members)
    build["rotation"] = {
        "seed": f"{arm}-v1:<row id>",
        "unrotated_families": sorted(UNROTATED & build["families"].keys()),
    }
    return write_arm(rows, out_dir, arm, build)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--out-dir", required=True, type=Path)
    for name in SOURCE_DIRS:
        parser.add_argument(f"--{name}", type=Path, metavar="DIR")
    for flag in CAP_FLAGS:
        parser.add_argument(f"--{flag.replace('_', '-')}", type=int, metavar="N")
    args = parser.parse_args(argv)
    needed = [source_dir(module) for module in ARMS[args.arm]]
    missing = [f"--{name}" for name in needed if getattr(args, name) is None]
    if missing:
        parser.error(f"--arm {args.arm} needs {' '.join(missing)}")
    foreign = [
        f"--{name}"
        for name in SOURCE_DIRS
        if name not in needed and getattr(args, name) is not None
    ]
    caps = {}
    for flag, key in CAP_FLAGS.items():
        value = getattr(args, flag)
        if value is None:
            continue
        if key not in default_caps(args.arm):
            foreign.append(f"--{flag.replace('_', '-')}")
        elif value < 0:
            parser.error(f"--{flag.replace('_', '-')} must be >= 0")
        else:
            caps[key] = value
    if foreign:
        parser.error(f"--arm {args.arm} does not use {' '.join(foreign)}")
    try:
        manifest = build_arm(
            args.arm, {name: getattr(args, name) for name in needed}, args.out_dir, caps
        )
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for warning in manifest["build"]["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(json.dumps(manifest, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
