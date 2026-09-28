"""Apply the A7 admission screens to pre-admission sub-arms and resolve views.

Inputs: a `v2.data.a7.build_a7` output directory, one private overlap receipt
per sub-arm (`v2.data.overlap` over its train and aho files), one shortcut
receipt per sub-arm (`v2.data.shortcut` over its train file) and a length file
from `v2.data.a7.lengths`. Rules (a7-prereg §2): whole-group overlap quarantine
for every role except `--report-only-role`; whole-group native budget; and, for
generated sub-arms only, dropping family x task-type cells whose state-removed
or option-only accuracy exceeds the cross-validated majority by the shortcut
margin. Writes final/<sub>.{train,aho}.jsonl, final/views/<view>.json and
final/admission.json (counts only, never text).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from v2.data.a7.build_a7 import (
    ALL_SUB_ARMS,
    SHORTCUT_GATED_SUB_ARMS,
    _json_bytes,
    _write_new,
    keyed,
)
from v2.data.apply_quarantine import overlap_groups
from v2.data.freeze import canonical_jsonl, parse_jsonl

GATED_VIEWS = ("state_removed", "option_only")
MIN_CELL_ROWS = 30
BUDGET_TOKENIZER = "qwen3.5-0.8b-base@dc7cdfe2"
MAX_NATIVE_TOKENS = 8192
KAI_TOKENIZER = "kai-0.6b@7185f514"
KAI_LIMIT = 1024


def shortcut_cells(receipt: Mapping[str, Any]) -> dict[str, list[str]]:
    """(family|task_type) -> gated views whose accuracy exceeds the margin."""
    cells: dict[str, list[str]] = collections.defaultdict(list)
    for view in GATED_VIEWS:
        by_task = receipt.get("views", {}).get(view, {}).get("by_task_family", {})
        for task_type, families in by_task.items():
            for family, summary in families.items():
                if summary["n"] >= MIN_CELL_ROWS and summary["exceeds_margin"]:
                    cells[f"{family}|{task_type}"].append(view)
    return dict(sorted(cells.items()))


def admit_sub_arm(
    name: str,
    parts: Mapping[str, Sequence[dict[str, Any]]],
    overlap_receipt: Mapping[str, Any],
    shortcut_receipt: Mapping[str, Any] | None,
    lengths: Mapping[str, Mapping[str, int]],
    report_only: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = [row for part in ("train", "aho") for row in parts.get(part, [])]
    quarantine = overlap_groups(overlap_receipt, report_only)
    missing = [row["id"] for row in rows if row["id"] not in lengths]
    if missing:
        raise ValueError(f"{name}: {len(missing)} rows lack native lengths")
    over_budget = {
        row["group_id"]
        for row in rows
        if lengths[row["id"]][BUDGET_TOKENIZER] > MAX_NATIVE_TOKENS
    }
    cells = (
        shortcut_cells(shortcut_receipt)
        if name in SHORTCUT_GATED_SUB_ARMS and shortcut_receipt is not None
        else {}
    )
    removed: dict[str, collections.Counter[tuple[str, str]]] = {
        reason: collections.Counter()
        for reason in ("overlap_quarantine", "native_budget", "shortcut_cell")
    }
    kept: dict[str, list[dict[str, Any]]] = {"train": [], "aho": []}
    for part in ("train", "aho"):
        for row in parts.get(part, []):
            cell = (row["family"], row["task_type"])
            if row["group_id"] in quarantine:
                removed["overlap_quarantine"][cell] += 1
            elif row["group_id"] in over_budget:
                removed["native_budget"][cell] += 1
            elif f"{row['family']}|{row['task_type']}" in cells:
                removed["shortcut_cell"][cell] += 1
            else:
                kept[part].append(row)
    roles: collections.Counter[str] = collections.Counter()
    for roles_of_group in quarantine.values():
        roles.update(roles_of_group)
    reported: collections.Counter[str] = collections.Counter()
    for record in overlap_receipt.get("groups", {}).values():
        reported.update(set(record.get("roles", [])) & report_only)
    final_rows = kept["train"] + kept["aho"]
    report = {
        "rows_in": {part: len(parts.get(part, [])) for part in ("train", "aho")},
        "rows_out": {part: len(kept[part]) for part in ("train", "aho")},
        "removed": {reason: keyed(counts) for reason, counts in removed.items()},
        "removed_totals": {
            reason: sum(counts.values()) for reason, counts in removed.items()
        },
        "quarantined_groups": len(quarantine),
        "quarantined_groups_by_role": dict(sorted(roles.items())),
        "report_only_groups_by_role": dict(sorted(reported.items())),
        "over_budget_groups": len(over_budget),
        "shortcut_cells_dropped": cells,
        "kai_over_1024_rows_flagged": sum(
            lengths[row["id"]].get(KAI_TOKENIZER, 0) > KAI_LIMIT for row in final_rows
        ),
    }
    return kept, report


def resolve_view(
    view: Mapping[str, Any], final_ids: Mapping[str, str]
) -> dict[str, Any]:
    members = [dict(item) for item in view["members"] if item["id"] in final_ids]
    for item in members:
        item["part"] = final_ids[item["id"]]
    return {
        **{key: view[key] for key in ("view", "source_file_sha256", "rows", "used_by")},
        "members": members,
        "covered": len(members),
        "removed_by_admission": view["covered"] - len(members),
        "uncovered_at_build": view["uncovered"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--overlap-receipt", action="append", default=[], help="SUB=path"
    )
    parser.add_argument(
        "--shortcut-receipt", action="append", default=[], help="SUB=path"
    )
    parser.add_argument("--lengths", type=Path, required=True)
    parser.add_argument("--report-only-role", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        parser.error(f"refusing to reuse {args.out_dir}")

    def named(values: list[str]) -> dict[str, Path]:
        pairs = dict(value.split("=", 1) for value in values)
        return {name: Path(path) for name, path in pairs.items()}

    overlaps = named(args.overlap_receipt)
    shortcuts = named(args.shortcut_receipt)
    lengths = {}
    with args.lengths.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            lengths[item["id"]] = item
    report_only = set(args.report_only_role)
    args.out_dir.mkdir(parents=True, mode=0o700)
    (args.out_dir / "views").mkdir(mode=0o700)
    reports: dict[str, Any] = {}
    outputs: dict[str, str] = {}
    final_ids: dict[str, str] = {}
    for name in ALL_SUB_ARMS:
        parts = {}
        for part in ("train", "aho"):
            path = args.build_dir / "prelim" / f"{name}.{part}.jsonl"
            parts[part] = (
                parse_jsonl(path.read_bytes(), str(path)) if path.exists() else []
            )
        if not any(parts.values()):
            continue
        if name not in overlaps:
            raise ValueError(f"{name}: an overlap receipt is required")
        overlap = json.loads(overlaps[name].read_text(encoding="utf-8"))
        shortcut = (
            json.loads(shortcuts[name].read_text(encoding="utf-8"))
            if name in shortcuts
            else None
        )
        if name in SHORTCUT_GATED_SUB_ARMS and shortcut is None:
            raise ValueError(f"{name}: generated sub-arms need a shortcut receipt")
        kept, report = admit_sub_arm(
            name, parts, overlap, shortcut, lengths, report_only
        )
        report["overlap_receipt_sha256"] = hashlib.sha256(
            overlaps[name].read_bytes()
        ).hexdigest()
        if name in shortcuts:
            report["shortcut_receipt_sha256"] = hashlib.sha256(
                shortcuts[name].read_bytes()
            ).hexdigest()
        reports[name] = report
        for part, rows in kept.items():
            for row in rows:
                final_ids[row["id"]] = part
            if rows:
                data = canonical_jsonl(rows)
                path = args.out_dir / f"{name}.{part}.jsonl"
                _write_new(path, data)
                outputs[path.name] = hashlib.sha256(data).hexdigest()
    for view_path in sorted((args.build_dir / "views").glob("*.prelim.json")):
        view = json.loads(view_path.read_text(encoding="utf-8"))
        resolved = resolve_view(view, final_ids)
        data = _json_bytes(resolved)
        path = args.out_dir / "views" / f"{view['view']}.json"
        _write_new(path, data)
        outputs[f"views/{path.name}"] = hashlib.sha256(data).hexdigest()
        reports.setdefault("views", {})[view["view"]] = {
            key: resolved[key]
            for key in ("covered", "removed_by_admission", "uncovered_at_build", "rows")
        }
    build_manifest = args.build_dir / "build-manifest.json"
    admission = {
        "schema": "decision2.v2.a7.admission.v1",
        "build_manifest_sha256": hashlib.sha256(
            build_manifest.read_bytes()
        ).hexdigest(),
        "lengths_sha256": hashlib.sha256(args.lengths.read_bytes()).hexdigest(),
        "report_only_roles": sorted(report_only),
        "rules": {
            "overlap": "whole-group quarantine for any non-report-only role",
            "native_budget": f"whole group if any row > {MAX_NATIVE_TOKENS} {BUDGET_TOKENIZER} tokens",
            "shortcut": f"generated sub-arms {sorted(SHORTCUT_GATED_SUB_ARMS)}: drop family|task cells "
            f"with n >= {MIN_CELL_ROWS} exceeding majority + margin on {list(GATED_VIEWS)}",
        },
        "sub_arms": reports,
        "outputs": outputs,
    }
    _write_new(args.out_dir / "admission.json", _json_bytes(admission))
    print(
        json.dumps(
            {
                name: report["rows_out"]
                for name, report in reports.items()
                if name != "views"
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
