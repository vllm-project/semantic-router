"""Make a 1,024-row legacy control for the frozen combined training budget.

The control shares every legacy row in combined_1024 and replaces its 171
programmatic/FLUTE rows with untouched legacy rows. Replacement counts match
task type and language exactly; exact tokenizer lengths guide row selection.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from training.data import build_budget_subsets as budgets
from training.data import build_pilot as pilot
from training.data.build_css_flute import read_panel_exclusions
from training.model.data import check_partition_isolation, load_partition

FROZEN_LEGACY_SHA256 = (
    "603600a6d1aeabe9179e5d3a85f275817f79a614e372bd64cb16584f50b83e17"
)
FROZEN_COMBINED_1024_SHA256 = (
    "7ab3df2a4e2c5ac74e12923b6ba94ef192ed90b94a8a612138ff9fe0cc8b1ddd"
)


def stratum(row: dict[str, Any]) -> tuple[str, str]:
    return row["task_type"], row["language"]


def candidate_rank(seed: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{seed}\0{row['id']}".encode()).hexdigest()


def nearest_length_rows(
    target: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    lengths: dict[str, int],
    seed: str,
) -> list[dict[str, Any]]:
    if len(candidates) < len(target):
        raise ValueError(
            f"Unmatchable task/language stratum: need {len(target)}, have {len(candidates)}"
        )
    ordered = sorted(
        (lengths[row["id"]], candidate_rank(seed, row), row["id"], row)
        for row in candidates
    )
    selected = []
    for row in sorted(target, key=lambda item: (lengths[item["id"]], item["id"])):
        desired = lengths[row["id"]]
        position = bisect.bisect_left(ordered, (desired, "", ""))
        positions = [
            index for index in (position - 1, position) if 0 <= index < len(ordered)
        ]
        best = min(
            positions,
            key=lambda index: (
                abs(ordered[index][0] - desired),
                ordered[index][1],
                ordered[index][2],
            ),
        )
        selected.append(ordered.pop(best)[3])
    return selected


def improve_token_total(
    selected: dict[tuple[str, str], list[dict[str, Any]]],
    available: dict[tuple[str, str], list[dict[str, Any]]],
    lengths: dict[str, int],
    target_total: int,
    seed: str,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    """Swap within strata only when it reduces the absolute total-token gap."""
    selected = {key: list(rows) for key, rows in selected.items()}
    selected_ids = {row["id"] for rows in selected.values() for row in rows}
    unused = {
        key: [row for row in rows if row["id"] not in selected_ids]
        for key, rows in available.items()
    }
    swaps = []
    for _ in range(32):
        current = sum(lengths[row["id"]] for rows in selected.values() for row in rows)
        residual = target_total - current
        if residual == 0:
            break
        best = None
        for key in selected:
            for selected_index, old in enumerate(selected[key]):
                for unused_index, new in enumerate(unused[key]):
                    delta = lengths[new["id"]] - lengths[old["id"]]
                    gap = abs(residual - delta)
                    if gap >= abs(residual):
                        continue
                    score = (
                        gap,
                        abs(delta),
                        candidate_rank(seed, new),
                        key,
                        old["id"],
                        selected_index,
                        unused_index,
                    )
                    if best is None or score < best[0]:
                        best = (score, key, selected_index, unused_index, delta)
        if best is None:
            break
        _, key, selected_index, unused_index, delta = best
        removed = selected[key][selected_index]
        inserted = unused[key][unused_index]
        selected[key][selected_index] = inserted
        unused[key][unused_index] = removed
        swaps.append(
            {
                "stratum": list(key),
                "removed_id": removed["id"],
                "inserted_id": inserted["id"],
                "token_delta": delta,
            }
        )
    return selected, swaps


def choose_replacements(
    target: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    lengths: dict[str, int],
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_target: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    by_candidate: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    for row in target:
        by_target[stratum(row)].append(row)
    for row in candidates:
        by_candidate[stratum(row)].append(row)
    selected = {}
    availability = []
    impossible = []
    for key, rows in sorted(by_target.items()):
        pool = by_candidate.get(key, [])
        report = {
            "task_type": key[0],
            "language": key[1],
            "target_rows": len(rows),
            "available_singletons": len(pool),
            "target_tokens": sum(lengths[row["id"]] for row in rows),
            "minimum_available_tokens": sum(
                sorted(lengths[row["id"]] for row in pool)[: len(rows)]
            ),
        }
        availability.append(report)
        if len(pool) < len(rows):
            impossible.append(report)
        else:
            selected[key] = nearest_length_rows(rows, pool, lengths, f"{seed}/{key}")
    if impossible:
        raise ValueError(f"Unmatchable task/language strata: {impossible}")
    target_tokens = sum(lengths[row["id"]] for row in target)
    initial_tokens = sum(
        lengths[row["id"]] for rows in selected.values() for row in rows
    )
    selected, swaps = improve_token_total(
        selected, by_candidate, lengths, target_tokens, seed
    )
    output = [row for rows in selected.values() for row in rows]
    if len(output) != len(target) or len({row["id"] for row in output}) != len(output):
        raise AssertionError("Replacement selection count or IDs differ")
    selected_counts = collections.Counter(stratum(row) for row in output)
    if selected_counts != collections.Counter(stratum(row) for row in target):
        raise AssertionError("Task/language distribution was not preserved")
    for item in availability:
        key = item["task_type"], item["language"]
        item["selected_tokens"] = sum(lengths[row["id"]] for row in selected[key])
        item["row_count_difference"] = len(selected[key]) - item["target_rows"]
        item["token_difference"] = item["selected_tokens"] - item["target_tokens"]
    return output, {
        "strata": availability,
        "unmatchable_strata": [],
        "initial_replacement_tokens": initial_tokens,
        "target_replacement_tokens": target_tokens,
        "selected_replacement_tokens": sum(lengths[row["id"]] for row in output),
        "token_swaps": swaps,
        "rank_seed": seed,
    }


def counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        name: dict(sorted(collections.Counter(row[name] for row in rows).items()))
        for name in ("source", "family", "task_type", "language")
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    legacy_file = args.legacy_source.resolve()
    combined_file = args.combined_source.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if pilot.sha_file(legacy_file) != FROZEN_LEGACY_SHA256:
        raise ValueError("Legacy source differs from the frozen 6k control")
    if pilot.sha_file(combined_file) != FROZEN_COMBINED_1024_SHA256:
        raise ValueError("Combined source differs from the frozen 1024 budget")
    legacy = load_partition(legacy_file, "train")
    combined = load_partition(combined_file, "train")
    select_file, cal_file = args.select_file.resolve(), args.cal_file.resolve()
    select = load_partition(select_file, "select")
    cal = load_partition(cal_file, "cal")
    if len(legacy) != 6000 or len(combined) != 1024:
        raise ValueError("Frozen source cardinality changed")
    old_by_id = {row["id"]: row for row in legacy}
    shared = [row for row in combined if budgets.bucket(row) == "legacy"]
    target = [row for row in combined if budgets.bucket(row) != "legacy"]
    if len(shared) != 853 or len(target) != 171:
        raise ValueError("Frozen 1024 source mix changed")
    if any(
        row["id"] not in old_by_id
        or pilot.canonical(old_by_id[row["id"]]) != pilot.canonical(row)
        for row in shared
    ):
        raise ValueError("Shared legacy core differs from the pure legacy source")
    group_sizes = collections.Counter(row["group_id"] for row in legacy)
    shared_ids = {row["id"] for row in shared}
    candidate = [
        row
        for row in legacy
        if row["id"] not in shared_ids and group_sizes[row["group_id"]] == 1
    ]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = {
        row["id"]: pilot.count_tokens(row, tokenizer) for row in [*legacy, *target]
    }
    replacement, match = choose_replacements(target, candidate, lengths, args.seed)
    control = shared + replacement
    control.sort(
        key=lambda row: (
            hashlib.sha256(f"{args.seed}\0output\0{row['id']}".encode()).hexdigest(),
            row["id"],
        )
    )
    if len(control) != 1024 or any(
        not row["source"].startswith("legacy:") for row in control
    ):
        raise AssertionError("Pure legacy control is not exactly 1024 old rows")
    if len({row["group_id"] for row in control}) > len(control):
        raise AssertionError("Invalid group count")
    # A selected group must be present in its entirety in the frozen source.
    selected_groups = collections.Counter(row["group_id"] for row in control)
    if any(selected_groups[group] != group_sizes[group] for group in selected_groups):
        raise AssertionError("Control split a legacy source group")
    check_partition_isolation({"train": control, "select": select, "cal": cal})
    cross = {
        "train_select": pilot.overlap_audit(control, select),
        "train_cal": pilot.overlap_audit(control, cal),
    }
    if any(
        pilot.audit_has_exact_overlap(audit) or audit["near_duplicate"]["count"]
        for audit in cross.values()
    ):
        raise ValueError("Pure legacy control overlaps independent SELECT/CAL")
    if pilot.train_consistency_audit(control)["conflicting_gold_groups"]:
        raise ValueError("Pure legacy control has conflicting gold")
    excluded, panel_receipt = read_panel_exclusions(args.panel_dir.resolve())
    panel = budgets.panel_overlap(control, excluded)
    control_tokens = sum(lengths[row["id"]] for row in control)
    combined_tokens = sum(lengths[row["id"]] for row in combined)
    if max(lengths[row["id"]] for row in control) > args.max_row_tokens:
        raise ValueError("Control has a row over the exact token cap")
    payloads = {
        "pure_legacy_1024.train.jsonl": pilot.jsonl_bytes(control),
        "select.jsonl": select_file.read_bytes(),
        "cal.jsonl": cal_file.read_bytes(),
    }
    output_dir.mkdir(parents=True, mode=0o700)
    for name, content in payloads.items():
        pilot._atomic_write(output_dir / name, content)
    actual_type_lang = collections.Counter(stratum(row) for row in control)
    target_type_lang = collections.Counter(stratum(row) for row in combined)
    if actual_type_lang != target_type_lang:
        raise AssertionError(
            "Pure legacy control does not match task/language joint counts"
        )
    manifest = {
        "schema_version": "decision2-pure-legacy-1024-control/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "input_sources": {
            "legacy_6k_sha256": FROZEN_LEGACY_SHA256,
            "combined_1024_sha256": FROZEN_COMBINED_1024_SHA256,
            "select_sha256": pilot.sha_file(select_file),
            "cal_sha256": pilot.sha_file(cal_file),
        },
        "panel_exclusions": panel_receipt,
        "seed": args.seed,
        "core": {
            "shared_legacy_rows": len(shared),
            "shared_id_sha256": pilot.sha_bytes(
                pilot.canonical(sorted(shared_ids)).encode()
            ),
        },
        "replacement": match,
        "target_counts": counts(combined),
        "control_counts": counts(control),
        "unmatchable_family_strata": {
            family: count for family, count in counts(target)["family"].items()
        },
        "remaining_differences": {
            "task_language_joint_count": 0,
            "total_tokens": control_tokens - combined_tokens,
            "total_token_percent": 100
            * (control_tokens - combined_tokens)
            / combined_tokens,
        },
        "token_audit": {
            "tokenizer_revision": args.tokenizer_revision,
            "max_row_tokens": args.max_row_tokens,
            "target_total": combined_tokens,
            "control_total": control_tokens,
            "control_maximum": max(lengths[row["id"]] for row in control),
        },
        "cross_partition_audit": cross,
        "css_panel_overlap": panel,
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(content),
                "bytes": len(content),
                "rows": len(content.splitlines()),
            }
            for name, content in payloads.items()
        },
        "interpretation": "Matched pilot control, not a single-factor causal test: 171 programmatic/FLUTE rows are replaced by selected short legacy rows while type/language and token budgets are matched.",
        "limitations": [
            "Length matching chooses unusually short legacy examples and changes legacy source/family mix.",
            "The frozen SELECT/CAL remain monitor splits and CSS FLUTE evaluation is supervised for the combined arm only.",
            "Approximate near-duplicate checks cannot prove semantic independence.",
        ],
    }
    pilot._atomic_write(
        output_dir / "pure_legacy_control.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--combined-source", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default="decision2-pure-legacy-1024-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "train_sha256": manifest["outputs"]["pure_legacy_1024.train.jsonl"][
                    "sha256"
                ],
                "target_tokens": manifest["token_audit"]["target_total"],
                "control_tokens": manifest["token_audit"]["control_total"],
                "token_percent": manifest["remaining_differences"][
                    "total_token_percent"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
