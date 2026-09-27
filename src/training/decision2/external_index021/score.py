"""0.2.1 scoring adapter over the hash-checked public Decision Index 0.2 kit.

The public kit is a pinned runtime dependency, rather than copied into this
repository. No result emitted here is a maintainer-issued leaderboard rank.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from .protocol import aggregate, chance_skill, spec
from .selection import Selection, select, verify_kit


class _Rows:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def rows(self, apply_exclusions: bool = False):
        yield from self._rows


def verified_suite(directory: str | Path):
    verify_kit()
    from decision_index.suite.io import Suite

    suite = Suite(directory, "0.2")
    report = suite.verify(strict=True)
    if not report.get("match") or not report.get("exclusions_match"):
        raise ValueError(
            "0.2 source rows, added rows, or common exclusions differ from the pinned kit"
        )
    if suite.edition["rows_sha256"] != spec()["suite"]["base_rows_sha256"]:
        raise ValueError("0.2 source corpus hash differs from the 0.2.1 Space")
    if suite.edition["added_sha256"] != spec()["suite"]["added_rows_sha256"]:
        raise ValueError("added-row hash differs from the 0.2.1 protocol")
    return suite


def selected_rows(
    suite, *, home_policy="first", home_keep_run_ids=None
) -> tuple[list[dict], Selection]:
    all_rows = list(suite.rows(apply_exclusions=True))
    choice = select(
        all_rows, home_policy=home_policy, home_keep_run_ids=home_keep_run_ids
    )
    return [r for r in all_rows if choice.keep(r["_evaluation"])], choice


def acos_review_f1(rows: list[dict], results: dict) -> dict:
    """Review-level set F1; failed/incomplete reviews contribute zero."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["_evaluation"]["group_id"]].append(row)
    if not groups:
        raise ValueError("ACOS rows are missing")
    values = []
    complete = 0
    for group in groups.values():
        ok = all(
            results.get(r["_evaluation"]["run_id"], {}).get("status") == "ok"
            for r in group
        )
        if not ok:
            values.append(0.0)
            continue
        complete += 1
        gold, pred = set(), set()
        for row in group:
            answers = results[row["_evaluation"]["run_id"]]["response"]["answers"]
            for key, question in row["questions"].items():
                if question["type"] != "choice":
                    raise ValueError(
                        "ACOS projection unexpectedly uses a non-Choice question"
                    )
                item = (row["_evaluation"]["run_id"], key)
                if row["expected"][key] == "yes":
                    gold.add(item)
                if answers[key]["choice"] == "yes":
                    pred.add(item)
        values.append(
            2 * len(gold & pred) / (len(gold) + len(pred)) if gold or pred else 1.0
        )
    raw = sum(values) / len(values)
    chance = spec()["chance"]["38"]
    return {
        "raw": raw,
        "skill": chance_skill(raw, chance),
        "coverage": complete / len(groups),
        "random": chance,
        "rule": "per-review F1",
        "review_count": len(groups),
        "complete_reviews": complete,
    }


def _kit_spec() -> dict:
    from decision_index.scoring import index02

    base = index02.spec()
    s = spec()
    base["edition"] = "0.2.1"
    base["panel_id"] = s["panel_id"]
    base["areas"] = [{"id": a["id"], "benchmarks": a["benchmarks"]} for a in s["areas"]]
    base["chance"] = {key: {"chance": value} for key, value in s["chance"].items()}
    base["not_in_index"] = [{"id": n} for n in s["not_in_index"]]
    if set(map(int, base["added"])) != set(s["added_ids"]):
        raise ValueError("added benchmark IDs differ from pinned 0.2 kit")
    return base


def score_rows(
    rows: list[dict], results: dict, *, selection: Selection | None = None
) -> dict:
    """Score a preselected 0.2.1 row set using the pinned 0.2 native metrics."""
    verify_kit()
    from decision_index.scoring import added, index02
    from decision_index.scoring import index as native_index
    from decision_index.scoring.report import benchmark_summary

    kit_spec = _kit_spec()
    added_ids = set(spec()["added_ids"])
    base = [r for r in rows if r["_evaluation"]["catalog_id"] not in added_ids]
    extras = collections.defaultdict(list)
    for r in rows:
        if r["_evaluation"]["catalog_id"] in added_ids:
            extras[r["_evaluation"]["catalog_id"]].append(r)
    summary = benchmark_summary(_Rows(base), results, "external-index-0.2.1", rows=base)
    native = {b["catalog_id"]: b for b in summary["benchmarks"]}
    native.update({n: added.report(n, rs, results) for n, rs in extras.items()})
    track_scored = native_index.score_panel(_Rows(base), results)
    acos = acos_review_f1(
        [r for r in base if r["_evaluation"]["catalog_id"] == 38], results
    )
    values = {}
    for area in spec()["areas"]:
        for number in area["benchmarks"]:
            if number == 38:
                value = acos
            else:
                if number not in native and number not in track_scored:
                    raise ValueError(f"missing native benchmark {number}")
                value = index02.benchmark_value(
                    number, kit_spec, track_scored.get(number), native.get(number)
                )
            values[number] = {
                key: value[key]
                for key in ("raw", "skill", "coverage", "random", "rule")
            }
    result = aggregate(values)
    counts = collections.Counter(
        results.get(r["_evaluation"]["run_id"], {}).get("status", "pending")
        for r in rows
    )
    completed = len(rows) - counts["pending"]
    result.update(
        benchmarks={str(k): v for k, v in sorted(values.items())},
        completed=completed,
        scoreable=len(rows),
        complete=completed == len(rows) and counts["error"] == 0,
        counts=dict(counts),
        selection=(selection.__dict__ | {"keep_run_ids": None}) if selection else None,
        claim="independent provisional 0.2.1 reproduction; Home row-copy identity is unverified",
    )
    # A row-selection keep list alone does not attest upstream equivalence.
    result["official_equivalent"] = False
    return result


def score_run(
    suite_dir: str | Path,
    results_path: str | Path,
    *,
    home_policy="first",
    home_keep_run_ids: set[str] | None = None,
    out: str | Path | None = None,
) -> dict:
    from decision_index.scoring.report import load_results

    suite = verified_suite(suite_dir)
    rows, selection = selected_rows(
        suite, home_policy=home_policy, home_keep_run_ids=home_keep_run_ids
    )
    results = load_results(results_path)
    scored = score_rows(rows, results, selection=selection)
    if out is not None:
        Path(out).write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
    return scored


def home_sensitivity(
    suite_dir: str | Path, results_path: str | Path, *, out: str | Path | None = None
) -> dict:
    """Score both permissible Home copy policies on the same full prediction file.

    The caller must have evaluated both members of all 24 duplicate pairs.
    This exposes the protocol ambiguity without pretending either policy is
    the unpublished upstream row selection.
    """
    from decision_index.scoring.report import load_results

    suite = verified_suite(suite_dir)
    results = load_results(results_path)
    scored = {}
    for policy in ("first", "last"):
        rows, selection = selected_rows(suite, home_policy=policy)
        scored[policy] = score_rows(rows, results, selection=selection)
        if not scored[policy]["complete"]:
            raise ValueError(
                f"{policy} Home policy has missing predictions or evaluator errors"
            )
    first = scored["first"]["scores"]["balanced_skill"]
    last = scored["last"]["scores"]["balanced_skill"]
    report = {
        "edition": "0.2.1",
        "claim": "provisional Home-copy sensitivity, not an official rank",
        "first": first,
        "last": last,
        "minimum": min(first, last),
        "maximum": max(first, last),
        "spread": abs(first - last),
        "first_selection_sha256": scored["first"]["selection"]["keep_ids_sha256"],
        "last_selection_sha256": scored["last"]["selection"]["keep_ids_sha256"],
    }
    if out is not None:
        Path(out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run(
    suite_dir: str | Path,
    *,
    engine: str,
    engine_options: dict,
    out_dir: str | Path,
    home_policy="first",
    home_keep_run_ids: set[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
) -> dict:
    """Run the public kit's native Engine contract on selected 0.2.1 rows."""
    from decision_index.runner import run as kit_run

    suite = verified_suite(suite_dir)
    _rows, selection = selected_rows(
        suite, home_policy=home_policy, home_keep_run_ids=home_keep_run_ids
    )
    return kit_run(
        engine,
        engine_options,
        suite.row_paths,
        out_dir,
        limit=limit,
        resume=resume,
        compact=True,
        corpus_sha256=spec()["suite"]["base_rows_sha256"],
        keep=selection.keep,
    )
