"""Build the verifiable Decision 2.0 arms (a2, a4 = a4h + a4r, a6) as JSONL plus a manifest.

Run from ``src/training/decision2``::

    python3 -m v2.data.verifiable.build --arm a2 --seed decision2-a2-v1 --out-dir /tmp/out
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

from training.model.data import canonical, file_sha256
from v2.data.verifiable import (
    a2_calendar,
    a2_counting,
    a2_injection,
    a2_multihop,
    a2_ordering,
    a2_units,
    a6_band_rubric,
    a6_checklist,
    a6_evidence_status,
    a6_interpolation,
    a6_rank_position,
    core,
)

A2_FAMILIES: tuple[ModuleType, ...] = (
    a2_calendar,
    a2_counting,
    a2_multihop,
    a2_injection,
    a2_ordering,
    a2_units,
)
A6_GENERAL: tuple[ModuleType, ...] = (
    a6_band_rubric,
    a6_checklist,
    a6_interpolation,
    a6_rank_position,
)
A6_FAMILIES: tuple[ModuleType, ...] = (*A6_GENERAL, a6_evidence_status)
LEVELS = tuple(range(2, 11))
HARMONIC = sum(Fraction(1, level) for level in LEVELS)
DEFAULT_GROUPS = {"a2": 375, "a4": 167}
DEFAULT_A6_ROWS_PER_LEVEL = 600
A4_NAMESPACE = "a4"


class BuildStats:
    def __init__(self) -> None:
        self.disagreements = 0
        self.dropped_rows = 0
        self.dropped_groups = 0
        self.identical_a4_distractors = 0


def a2_task(index: int) -> str:
    return "choice" if index % 3 == 0 else "noul"


def language(index: int, zh_share: Fraction) -> str:
    return "zh" if core.is_zh(index, zh_share) else "en"


def rows_per_level(groups_per_family: int | None) -> int:
    if groups_per_family is None:
        return DEFAULT_A6_ROWS_PER_LEVEL
    return max(1, round(Fraction(5 * groups_per_family) / HARMONIC))


def a6_plan(per_level: int) -> dict[str, list[int]]:
    """Level count of every group, per family; rows per level count are ~``per_level``."""
    plan: dict[str, list[int]] = {module.FAMILY: [] for module in A6_FAMILIES}
    for levels in LEVELS:
        groups = max(1, round(Fraction(per_level, levels)))
        if levels == 3:
            evidence = round(Fraction(groups, 2))
            plan[a6_evidence_status.FAMILY] += [3] * evidence
            groups -= evidence
        for j in range(groups):
            plan[A6_GENERAL[(j + levels) % len(A6_GENERAL)].FAMILY].append(levels)
    return plan


def _group_id(arm: str, family: str, lang: str, seed: str, index: int) -> str:
    return f"{arm}:{family}:{lang}:{core.group_hash(seed, family, lang, index)}"


def counterfactual_rows(
    arm: str,
    module: ModuleType,
    lang: str,
    index: int,
    seed: str,
    task: str,
    levels: int | None,
    stats: BuildStats,
) -> list[dict[str, Any]]:
    family = module.FAMILY
    rng = core.make_rng(seed, family, lang, index)
    group = module.build_group(rng, lang, task, levels)
    core.check_group(group)
    group_id = _group_id(arm, family, lang, seed, index)
    split = "select" if core.held_out(group_id) else "train"
    injected = group.meta.get("injected_option")
    order = core.make_rng(seed, family, lang, index, "variant-order").sample(
        group.variants, len(group.variants)
    )
    rows, bad = [], 0
    for variant_index, variant in enumerate(order):
        if task == "choice" and not core.presence_ok(
            variant.state, group.options, [injected] if injected else []
        ):
            raise AssertionError(
                f"{family} {index}: option strings leak through presence in the state"
            )
        reparsed = module.reparse(
            variant.state, group.instructions, group.options, lang
        )
        bad += reparsed != variant.label
        meta = {"subtype": group.subtype, **group.meta}
        rows.append(
            core.make_row(
                arm=arm,
                family=family,
                lang=lang,
                index=index,
                variant_index=variant_index,
                group_id=group_id,
                state=variant.state,
                instructions=group.instructions,
                options=group.options,
                label=variant.label,
                task_type=group.task_type,
                template=group.template,
                edit=variant.edit,
                facts=variant.facts,
                reparse_label=reparsed,
                meta=meta,
                split=split,
            )
        )
    if bad:
        stats.disagreements += bad
        stats.dropped_rows += len(rows)
        stats.dropped_groups += 1
        return []
    return rows


def build_a2(
    seed: str, groups_per_family: int, zh_share: Fraction, stats: BuildStats
) -> list[dict[str, Any]]:
    rows = []
    for module in A2_FAMILIES:
        for index in range(groups_per_family):
            rows += counterfactual_rows(
                "a2",
                module,
                language(index, zh_share),
                index,
                seed,
                a2_task(index),
                None,
                stats,
            )
    return rows


def build_a6(
    seed: str, per_level: int, zh_share: Fraction, stats: BuildStats
) -> list[dict[str, Any]]:
    rows = []
    plan = a6_plan(per_level)
    for module in A6_FAMILIES:
        for index, levels in enumerate(plan[module.FAMILY]):
            rows += counterfactual_rows(
                "a6",
                module,
                language(index, zh_share),
                index,
                seed,
                "score",
                levels,
                stats,
            )
    return rows


def _feasible_rank(
    rank: int, k: int, below: Sequence[str], above: Sequence[str]
) -> int:
    low = max(0, k - 1 - len(above))
    high = min(len(below), k - 1)
    return min(max(rank, low), high)


def _distractors(
    scenario: core.A4Scenario, rank: int, rng: Any
) -> tuple[list[str], list[str], int]:
    k = scenario.k
    if scenario.near_ordered:
        rank = _feasible_rank(rank, k, scenario.near_below, scenario.near_above)
        near = scenario.near_below[:rank] + scenario.near_above[: k - 1 - rank]
    else:
        near = scenario.near_above[: k - 1]
    for _ in range(64):
        if scenario.pool_ordered:
            r = _feasible_rank(rank, k, scenario.pool_below, scenario.pool_above)
            random_pick = rng.sample(scenario.pool_below, r) + rng.sample(
                scenario.pool_above, k - 1 - r
            )
        else:
            random_pick = rng.sample(scenario.pool_below + scenario.pool_above, k - 1)
        if set(random_pick) != set(near):
            break
    return near, random_pick, rank


def a4_rows(
    module: ModuleType,
    lang: str,
    index: int,
    seed: str,
    counters: collections.Counter,
    stats: BuildStats,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family = module.FAMILY
    ns_seed = f"{seed}/{A4_NAMESPACE}"
    scenario: core.A4Scenario = module.a4_scenario(
        core.make_rng(ns_seed, family, lang, index), lang
    )
    pick = core.make_rng(ns_seed, family, lang, index, "distractors")
    group_id = _group_id(A4_NAMESPACE, family, lang, ns_seed, index)
    split = "select" if core.held_out(group_id) else "train"
    k = scenario.k
    turn = counters[(family, "choice", k)]
    counters[(family, "choice", k)] += 1
    position, rank = turn % k, (turn // k) % k
    near, far, rank = _distractors(scenario, rank, pick)
    if set(near) == set(far):
        stats.identical_a4_distractors += 1
    slots = [i for i in range(k) if i != position]
    near_order, far_order = pick.sample(near, len(near)), pick.sample(far, len(far))
    truth = counters[(family, "noul")] % 2
    counters[(family, "noul")] += 1
    far_noul = pick.choice(scenario.noul_far)
    out: dict[str, list[dict[str, Any]]] = {"a4h": [], "a4r": []}
    bad = 0
    for arm, distractors, false_question in (
        ("a4h", near_order, scenario.noul_near),
        ("a4r", far_order, far_noul),
    ):
        descriptions = [""] * k
        descriptions[position] = scenario.gold
        for slot, text in zip(slots, distractors):
            descriptions[slot] = text
        if len(set(descriptions)) != k:
            raise AssertionError(f"{family} {index}: duplicate option text")
        base_meta = {"subtype": scenario.subtype, "a4_file": arm, "gold_rank": rank}
        for variant_index, (task, instructions, options, label) in enumerate(
            (
                (
                    "choice",
                    scenario.question,
                    core.choice_options(descriptions),
                    position,
                ),
                (
                    "noul",
                    scenario.noul_true if truth else false_question,
                    core.noul_options(lang),
                    truth,
                ),
            )
        ):
            reparsed = module.reparse(scenario.state, instructions, options, lang)
            bad += reparsed != label
            out[arm].append(
                core.make_row(
                    arm=arm,
                    family=family,
                    lang=lang,
                    index=index,
                    variant_index=variant_index,
                    group_id=group_id,
                    state=scenario.state,
                    instructions=instructions,
                    options=options,
                    label=label,
                    task_type=task,
                    template=f"{scenario.template}_{task}",
                    edit="none: single hard-negative scenario",
                    facts=scenario.facts,
                    reparse_label=reparsed,
                    meta={
                        **base_meta,
                        "distractors": "near" if arm == "a4h" else "random",
                    },
                    split=split,
                )
            )
    if bad:
        stats.disagreements += bad
        stats.dropped_rows += len(out["a4h"]) + len(out["a4r"])
        stats.dropped_groups += 1
        return [], []
    return out["a4h"], out["a4r"]


def build_a4(
    seed: str, groups_per_family: int, zh_share: Fraction, stats: BuildStats
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard, rand = [], []
    counters: collections.Counter = collections.Counter()
    for module in A2_FAMILIES:
        for index in range(groups_per_family):
            h, r = a4_rows(
                module, language(index, zh_share), index, seed, counters, stats
            )
            hard += h
            rand += r
    return hard, rand


# ---------------------------------------------------------------- manifest


def _quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {
        "min": ordered[0],
        "p10": pick(0.1),
        "median": pick(0.5),
        "p90": pick(0.9),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = collections.Counter
    by = lambda key: dict(sorted(count(key(r) for r in rows).items()))
    levels = lambda r: len(r["options"]) if r["task_type"] == "score" else None
    noul = [r for r in rows if r["task_type"] == "noul"]
    return {
        "rows": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "by_family": by(lambda r: r["family"]),
        "by_language": by(lambda r: r["language"]),
        "by_task_type": by(lambda r: r["task_type"]),
        "by_family_task_type": by(lambda r: f"{r['family']}/{r['task_type']}"),
        "by_label": by(lambda r: f"{r['task_type']}:{r['label']}"),
        "score_rows_by_levels": (
            by(lambda r: str(levels(r)))
            if any(r["task_type"] == "score" for r in rows)
            else {}
        ),
        "score_rows_by_levels_label": (
            by(lambda r: f"L{levels(r)}:{r['label']}")
            if any(r["task_type"] == "score" for r in rows)
            else {}
        ),
        "choice_rows_by_option_count_position": (
            by(lambda r: f"K{len(r['options'])}:{r['label']}")
            if any(r["task_type"] == "choice" for r in rows)
            else {}
        ),
        "noul_true_share": (
            round(sum(r["label"] for r in noul) / len(noul), 4) if noul else None
        ),
        "state_words": _quantiles(
            [core.words(r["state"], r["language"]) for r in rows]
        ),
        "state_chars": _quantiles([len(r["state"]) for r in rows]),
        "row_chars": _quantiles(
            [
                len(canonical({k: r[k] for k in ("state", "instructions", "options")}))
                for r in rows
            ]
        ),
    }


def code_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}


def content_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["id"]):
        digest.update((canonical(row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def generate(
    arm: str, seed: str, groups_per_family: int | None, zh_share: float
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    share = Fraction(str(zh_share))
    stats = BuildStats()
    parameters: dict[str, Any] = {
        "zh_share": zh_share,
        "groups_per_family": groups_per_family,
    }
    if arm == "a2":
        groups = groups_per_family or DEFAULT_GROUPS["a2"]
        parameters["effective_groups_per_family"] = groups
        rows = build_a2(seed, groups, share, stats)
        files = {"a2": rows}
    elif arm == "a6":
        per_level = rows_per_level(groups_per_family)
        parameters["rows_per_level_target"] = per_level
        parameters["groups_by_family_levels"] = {
            f: dict(sorted(collections.Counter(v).items()))
            for f, v in a6_plan(per_level).items()
        }
        files = {"a6": build_a6(seed, per_level, share, stats)}
    elif arm == "a4":
        groups = groups_per_family or DEFAULT_GROUPS["a4"]
        parameters["effective_groups_per_family"] = groups
        parameters["seed_namespace"] = f"{seed}/{A4_NAMESPACE}"
        hard, rand = build_a4(seed, groups, share, stats)
        files = {"a4h": hard, "a4r": rand}
    else:
        raise ValueError(f"unknown arm {arm}")
    split_files: dict[str, list[dict[str, Any]]] = {}
    for name, rows in files.items():
        split_files[f"{name}.train.jsonl"] = [r for r in rows if r["split"] == "train"]
        split_files[f"{name}.aho.jsonl"] = [r for r in rows if r["split"] == "select"]
    manifest = {
        "generator": core.GENERATOR,
        "version": core.VERSION,
        "arm": arm,
        "seed": seed,
        "parameters": parameters,
        "code_sha256": code_hashes(),
        "files": {
            name: {
                "rows": len(rows),
                "content_sha256": content_sha256(rows),
                "summary": summarize(rows),
            }
            for name, rows in split_files.items()
        },
        "oracle_disagreements": stats.disagreements,
        "dropped_rows": stats.dropped_rows,
        "dropped_groups": stats.dropped_groups,
        "identical_a4_distractor_sets": stats.identical_a4_distractors,
    }
    return split_files, manifest


def write(
    out_dir: Path,
    arm: str,
    files: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
) -> list[Path]:
    targets = [out_dir / name for name in files] + [out_dir / f"{arm}.build.json"]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {', '.join(existing)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        with (out_dir / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical(row) + "\n")
        manifest["files"][name]["file_sha256"] = file_sha256(out_dir / name)
    with (out_dir / f"{arm}.build.json").open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    return targets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=("a2", "a4", "a6"), required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--groups-per-family",
        type=int,
        help="a2/a4: groups per family (default 375 / 167). a6: sets rows per level count to "
        "round(5N / H) with H = sum 1/L over L=2..10, i.e. about 5N groups (default 600 rows per level count).",
    )
    parser.add_argument("--zh-share", type=float, default=0.3)
    args = parser.parse_args(argv)
    if args.groups_per_family is not None and args.groups_per_family < 1:
        parser.error("--groups-per-family must be positive")
    if not 0 <= args.zh_share <= 1:
        parser.error("--zh-share must be within [0, 1]")
    targets = [args.out_dir / f"{args.arm}.build.json"]
    names = ("a4h", "a4r") if args.arm == "a4" else (args.arm,)
    targets += [
        args.out_dir / f"{n}.{part}.jsonl" for n in names for part in ("train", "aho")
    ]
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        print(f"refusing to overwrite: {', '.join(existing)}", file=sys.stderr)
        return 2
    files, manifest = generate(
        args.arm, args.seed, args.groups_per_family, args.zh_share
    )
    write(args.out_dir, args.arm, files, manifest)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "files": {
                    name: info["rows"] for name, info in manifest["files"].items()
                },
                "oracle_disagreements": manifest["oracle_disagreements"],
                "dropped_rows": manifest["dropped_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
