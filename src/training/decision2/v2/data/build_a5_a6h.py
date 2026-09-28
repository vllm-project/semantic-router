"""Build the A5 (KLUE/JGLUE Choice and Noul) or A6h (human ordinal Score) arm.

    python3 -m v2.data.build_a5_a6h --arm a5 --klue DIR --jglue DIR --out-dir OUT
    python3 -m v2.data.build_a5_a6h --arm a6h --klue DIR --jglue DIR --argq DIR \\
        --saf-en DIR --saf-de DIR --out-dir OUT

Rules: data-arms-v1-prereg-amendment-1-2026-09-28.md sections 1-3. Only TRAIN
files are read. Caps keep whole groups (common.cap_groups); shortfalls are
recorded, not filled. Refuses to overwrite an existing arm.
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import training.model.data as contract
from training.model.data import file_sha256
from v2.data import textnorm
from v2.data.sources import argq, common, jglue, klue, ordinal, saf
from v2.data.sources.common import cap_groups, write_arm

SCHEMA = "decision2-a5-a6h-build/v1"
RULES = "data-arms-v1-prereg-amendment-1-2026-09-28.md sections 1-3"
ROOT = Path(__file__).resolve().parents[2]
Loader = Callable[[Path], tuple[list[dict[str, Any]], dict[str, Any]]]

A5_PLAN: tuple[tuple[str, str, str, Loader, int, str], ...] = (
    ("klue_ynat", "klue_ynat_train", "klue", klue.ynat, 1500, "a5-klue-ynat-v1"),
    ("klue_nli", "klue_nli_train", "klue", klue.nli, 1500, "a5-klue-nli-v1"),
    ("klue_mrc_answerable", "klue_mrc_train", "klue", klue.mrc, 1000, "a5-klue-mrc-v1"),
    (
        "jglue_jcommonsenseqa",
        "jglue_jcommonsenseqa_v1.3_train",
        "jglue",
        jglue.jcommonsenseqa,
        1000,
        "a5-jglue-jcqa-v1",
    ),
    (
        "jglue_jnli",
        "jglue_jnli_v1.3_train",
        "jglue",
        jglue.jnli,
        1000,
        "a5-jglue-jnli-v1",
    ),
)
EXACT_NOUL_BALANCE = frozenset({"klue_mrc_answerable"})
CLASS_BALANCE: dict[str, tuple[str, ...]] = {
    "klue_ynat": tuple(str(label) for label in range(len(klue.YNAT_SECTIONS))),
    "klue_nli": klue.NLI_LABELS,
    "jglue_jnli": klue.NLI_LABELS,
}
A5_GROUP_KEYS = {
    "klue_ynat": "guid",
    "klue_nli": "normalized premise; premises of mutually reversed pairs share the "
    "smallest normalized premise",
    "klue_mrc_answerable": "normalized context",
    "jglue_jcommonsenseqa": "q_id",
    "jglue_jnli": "image id (yjcaptions_id before the first '-'); images linked by "
    "mutually reversed pairs share the smallest image id",
}

SCORE_CAP = 1800
A6H_PLAN: tuple[tuple[str, str, str, str, Loader], ...] = (
    ("klue_sts_train", "klue_sts", "ko", "klue", klue.sts),
    ("jglue_jsts_v1.3_train", "jglue_jsts", "ja", "jglue", jglue.jsts),
    ("argq30k_train", "argq30k", "en", "argq", argq.argq),
    ("saf_en_train", "saf", "en", "saf_en", functools.partial(saf.saf, language="en")),
    ("saf_de_train", "saf", "de", "saf_de", functools.partial(saf.saf, language="de")),
)
A6H_GROUP_KEYS = {
    "klue_sts": "guid",
    "jglue_jsts": "image id (yjcaptions_id before the first '-')",
    "argq30k": "normalized argument",
    "saf": "normalized question + '|' + normalized provided_answer",
}


def module_hashes() -> dict[str, str]:
    modules = (
        Path(__file__),
        Path(common.__file__),
        Path(klue.__file__),
        Path(jglue.__file__),
        Path(argq.__file__),
        Path(saf.__file__),
        Path(ordinal.__file__),
        Path(textnorm.__file__),
        Path(contract.__file__),
    )
    return {
        path.resolve().relative_to(ROOT).as_posix(): file_sha256(path)
        for path in sorted(modules, key=lambda path: path.resolve().as_posix())
    }


def histogram(values: Iterable[Any]) -> dict[str, int]:
    return {
        str(key): count for key, count in sorted(collections.Counter(values).items())
    }


def level_table(
    rows: Sequence[Mapping[str, Any]], sizes: Sequence[int]
) -> dict[str, dict[str, int]]:
    table: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    for row in rows:
        table[len(row["options"])][row["label"]] += 1
    return {
        f"L{size}": {str(level): table[size][level] for level in range(size)}
        for size in sizes
    }


def strata_table(
    rows: Sequence[Mapping[str, Any]], sizes: Sequence[int], strata: Sequence[str]
) -> dict[str, dict[str, dict[str, int]]]:
    table: collections.Counter[tuple[int, int, str]] = collections.Counter(
        (len(row["options"]), row["label"], row["audit_metadata"]["stratum"])
        for row in rows
    )
    return {
        f"L{size}": {
            str(level): {stratum: table[size, level, stratum] for stratum in strata}
            for level in range(size)
        }
        for size in sizes
    }


def score_cell(row: Mapping[str, Any]) -> str:
    return f"{row['source']}|L{len(row['options'])}"


def class_index(row: Mapping[str, Any]) -> int:
    return CLASS_BALANCE[row["family"]].index(
        str(row["audit_metadata"]["original_label"])
    )


def _label(row: Mapping[str, Any]) -> int:
    return row["label"]


def _size(row: Mapping[str, Any]) -> int:
    return len(row["options"])


def _id(row: Mapping[str, Any]) -> str:
    return row["id"]


def load(
    loader: Loader, root: Path, source: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, report = loader(root)
    repeated = sorted(
        key for key, count in collections.Counter(map(_id, rows)).items() if count > 1
    )
    if repeated:
        raise ValueError(f"{source}: duplicate source ids behind rows {repeated[:5]}")
    return rows, report


def build_a5(dirs: Mapping[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    families: dict[str, Any] = {}
    for family, source, key, loader, cap, seed in A5_PLAN:
        candidates, report = load(loader, dirs[key], source)
        inputs[source] = report.pop("input")
        capped = cap_groups(candidates, cap, seed)
        entry: dict[str, Any] = {
            "source": source,
            **report,
            "cap": cap,
            "cap_seed": seed,
            "candidates": len(candidates),
            "after_cap": len(capped),
        }
        final = capped
        if family in EXACT_NOUL_BALANCE:
            final, cells = ordinal.balance(
                capped,
                cell=lambda row: row["family"],
                level=_label,
                levels=_size,
                ident=_id,
                seed=f"{seed}:balance",
                ratio=Fraction(1),
            )
            entry["noul_balance"] = {
                "rule": "exact 50/50 by hash-ordered downsampling of the capped set",
                "seed": f"{seed}:balance",
                **cells.get(family, {"before": [0, 0], "after": [0, 0], "limit": 0}),
            }
        if family in CLASS_BALANCE:
            classes = CLASS_BALANCE[family]
            final, cells = ordinal.balance(
                capped,
                cell=lambda row: row["family"],
                level=class_index,
                levels=lambda row: len(CLASS_BALANCE[row["family"]]),
                ident=_id,
                seed=f"{seed}:class-balance",
                ratio=Fraction(1),
            )
            cell = cells.get(family, {"before": [0] * len(classes), "limit": 0})
            entry["class_balance"] = {
                "rule": "equal rows per gold class by hash-ordered downsampling of "
                "the capped set",
                "seed": f"{seed}:class-balance",
                "classes": list(classes),
                "before": dict(zip(classes, cell["before"])),
                "after": dict(zip(classes, cell.get("after", [0] * len(classes)))),
                "per_class": cell["limit"],
            }
        entry.update(
            rows=len(final),
            groups=len({row["group_id"] for row in final}),
            shortfall=max(0, cap - len(final)),
            label_histogram=histogram(row["label"] for row in final),
            final_class_histogram=histogram(
                row["audit_metadata"].get("original_label", row["label"])
                for row in final
            ),
        )
        families[family] = entry
        rows.extend(final)
    return rows, {
        "schema": SCHEMA,
        "arm": "a5",
        "rules": RULES,
        "inputs": inputs,
        "modules": module_hashes(),
        "rotation": "common.rotate(choice_options(descriptions), gold, "
        "seed=f'{arm}-v1:{row_id}') with row_id the final make_row id",
        "group_keys": A5_GROUP_KEYS,
        "normalization": "v2.data.textnorm.normalize (NFKC, casefold, whitespace)",
        "families": families,
    }


def build_a6h(dirs: Mapping[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for source, family, language, key, loader in A6H_PLAN:
        candidates, report = load(loader, dirs[key], source)
        inputs[source] = report.pop("input")
        seed = f"a6h-{source}-v1"
        balanced, cells = ordinal.balance(
            candidates,
            cell=score_cell,
            level=_label,
            levels=_size,
            ident=_id,
            seed=f"{seed}:balance",
        )
        capped = cap_groups(balanced, SCORE_CAP, seed)
        sizes = report["level_counts"]
        sources[source] = {
            "family": family,
            "language": language,
            **report,
            "balance_rule": "per cell (source, L) keep at most "
            "floor(1.2 * rarest level) rows per level, hash-ordered",
            "balance_seed": f"{seed}:balance",
            "balance_cells": cells,
            "empty_level_cells": [
                key for key, cell in cells.items() if not cell["limit"]
            ],
            "levels_before_balance": level_table(candidates, sizes),
            "levels_after_balance": level_table(balanced, sizes),
            "cap": SCORE_CAP,
            "cap_seed": seed,
            "levels_after_cap": level_table(capped, sizes),
            "rows": len(capped),
            "groups": len({row["group_id"] for row in capped}),
            "shortfall": max(0, SCORE_CAP - len(capped)),
        }
        if "strata" in report:
            strata = list(report["strata"])
            sources[source]["strata_by_level_before_balance"] = strata_table(
                candidates, sizes, strata
            )
            sources[source]["strata_by_level_after_balance"] = strata_table(
                balanced, sizes, strata
            )
        rows.extend(capped)
    return rows, {
        "schema": SCHEMA,
        "arm": "a6h",
        "rules": RULES,
        "inputs": inputs,
        "modules": module_hashes(),
        "group_keys": A6H_GROUP_KEYS,
        "normalization": "v2.data.textnorm.normalize (NFKC, casefold, whitespace)",
        "label_histograms": {
            family: level_table(
                [row for row in rows if row["family"] == family],
                sorted({_size(row) for row in rows if row["family"] == family}),
            )
            for family in sorted({row["family"] for row in rows})
        },
        "sources": sources,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=("a5", "a6h"))
    parser.add_argument("--klue", type=Path, required=True)
    parser.add_argument("--jglue", type=Path, required=True)
    parser.add_argument("--argq", type=Path)
    parser.add_argument("--saf-en", type=Path)
    parser.add_argument("--saf-de", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    extra = {"argq": args.argq, "saf_en": args.saf_en, "saf_de": args.saf_de}
    if args.arm == "a6h" and not all(extra.values()):
        parser.error("--arm a6h needs --argq, --saf-en and --saf-de")
    if args.arm == "a5" and any(extra.values()):
        parser.error("--arm a5 takes only --klue and --jglue")
    existing = [
        name
        for name in (
            f"{args.arm}.train.jsonl",
            f"{args.arm}.aho.jsonl",
            f"{args.arm}.build.json",
        )
        if (args.out_dir / name).exists()
    ]
    if existing:
        parser.error(f"refusing to overwrite {existing} in {args.out_dir}")
    dirs = {"klue": args.klue, "jglue": args.jglue, **extra}
    rows, build = (build_a5 if args.arm == "a5" else build_a6h)(dirs)
    manifest = write_arm(rows, args.out_dir, args.arm, build)
    print(
        json.dumps(
            {
                part: {
                    key: manifest[part][key]
                    for key in ("rows", "groups", "sha256", "source")
                }
                for part in ("train", "aho")
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
