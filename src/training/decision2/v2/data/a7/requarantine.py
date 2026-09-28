"""Re-freeze admitted A7 sub-arms after later protected-overlap receipts.

Reads an admitted A7 run (`final/<sub>.{train,aho}.jsonl`, `final/views/`,
`final/admission.json`), removes every group flagged by the given private
lexical-overlap receipts (any role except `--report-only-role`) or listed at or
above the quarantine threshold of the given embedding receipts, and writes the
same layout to a new run directory: rows keep canonical order, views drop the
removed members, and `final/admission.json` carries the earlier admission plus a
`requarantine` section (counts by sub-arm, role and family; never text or ids).
Groups are removed from every sub-arm, so a flagged group cannot survive in
another file. The build manifest and pre-admission receipts are copied so the
new run can be frozen, isolated, assembled and uploaded by `run_nodeB.sh`.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from v2.data.a7.build_a7 import ALL_SUB_ARMS, _json_bytes, _write_new, keyed
from v2.data.apply_quarantine import embed_groups, overlap_groups
from v2.data.freeze import canonical_jsonl, parse_jsonl


def flagged_groups(
    overlap_receipts: Sequence[Mapping[str, Any]],
    embed_receipts: Sequence[Mapping[str, Any]],
    report_only: set[str],
) -> dict[str, dict[str, set[str]]]:
    """group_id -> {"lexical": roles, "embedding": roles}."""
    flagged: dict[str, dict[str, set[str]]] = collections.defaultdict(
        lambda: {"lexical": set(), "embedding": set()}
    )
    for receipt in overlap_receipts:
        for group, roles in overlap_groups(receipt, report_only).items():
            flagged[group]["lexical"].update(roles)
    for receipt in embed_receipts:
        for group, roles in embed_groups(receipt).items():
            flagged[group]["embedding"].update(roles)
    return dict(flagged)


def requarantine(
    parts: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    flagged: Mapping[str, Mapping[str, set[str]]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    kept: dict[str, dict[str, list[dict[str, Any]]]] = {}
    report: dict[str, Any] = {}
    for name, by_part in parts.items():
        kept[name] = {}
        removed_cells: collections.Counter[tuple[str, str]] = collections.Counter()
        removed_groups: set[str] = set()
        rows_in, rows_out = {}, {}
        for part, rows in by_part.items():
            kept[name][part] = [row for row in rows if row["group_id"] not in flagged]
            for row in rows:
                if row["group_id"] in flagged:
                    removed_cells[(row["family"], row["task_type"])] += 1
                    removed_groups.add(row["group_id"])
            rows_in[part], rows_out[part] = len(rows), len(kept[name][part])
        by_method_role: collections.Counter[str] = collections.Counter()
        for group in removed_groups:
            for method, roles in flagged[group].items():
                for role in roles:
                    by_method_role[f"{method}|{role}"] += 1
        report[name] = {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "groups_removed": len(removed_groups),
            "rows_removed_by_family_type": keyed(removed_cells),
            "groups_removed_by_method_role": dict(sorted(by_method_role.items())),
        }
    return kept, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--overlap-receipt", type=Path, action="append", default=[])
    parser.add_argument("--embed-receipt", type=Path, action="append", default=[])
    parser.add_argument("--report-only-role", action="append", default=[])
    parser.add_argument("--version", required=True)
    parser.add_argument("--out-run", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.out_run / "final").exists():
        parser.error(f"refusing to reuse {args.out_run / 'final'}")
    source = args.from_run / "final"
    receipts = {
        "overlap": [(path, path.read_bytes()) for path in args.overlap_receipt],
        "embedding": [(path, path.read_bytes()) for path in args.embed_receipt],
    }
    flagged = flagged_groups(
        [json.loads(data) for _, data in receipts["overlap"]],
        [json.loads(data) for _, data in receipts["embedding"]],
        set(args.report_only_role),
    )
    parts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name in ALL_SUB_ARMS:
        found = {}
        for part in ("train", "aho"):
            path = source / f"{name}.{part}.jsonl"
            if path.exists():
                found[part] = parse_jsonl(path.read_bytes(), str(path))
        if found:
            parts[name] = found
    kept, report = requarantine(parts, flagged)

    final = args.out_run / "final"
    (final / "views").mkdir(parents=True, mode=0o700)
    outputs: dict[str, str] = {}
    final_ids: dict[str, str] = {}
    for name, by_part in kept.items():
        for part, rows in by_part.items():
            for row in rows:
                final_ids[row["id"]] = part
            if rows:
                data = canonical_jsonl(rows)
                path = final / f"{name}.{part}.jsonl"
                _write_new(path, data)
                outputs[path.name] = hashlib.sha256(data).hexdigest()
    views_report = {}
    for view_path in sorted((source / "views").glob("*.json")):
        view = json.loads(view_path.read_text(encoding="utf-8"))
        members = [item for item in view["members"] if item["id"] in final_ids]
        removed = len(view["members"]) - len(members)
        view = {
            **view,
            "members": members,
            "covered": len(members),
            "removed_by_requarantine": removed,
        }
        data = _json_bytes(view)
        path = final / "views" / view_path.name
        _write_new(path, data)
        outputs[f"views/{path.name}"] = hashlib.sha256(data).hexdigest()
        views_report[view["view"]] = {
            "covered": len(members),
            "removed_by_requarantine": removed,
        }
    admission = json.loads((source / "admission.json").read_text(encoding="utf-8"))
    admission["version"] = args.version
    admission["requarantine"] = {
        "from_run_admission_sha256": hashlib.sha256(
            (source / "admission.json").read_bytes()
        ).hexdigest(),
        "receipts": {
            kind: [hashlib.sha256(data).hexdigest() for _, data in items]
            for kind, items in receipts.items()
        },
        "report_only_roles": sorted(args.report_only_role),
        "rule": "whole-group removal from every sub-arm for any lexical hit outside "
        "the report-only roles or any embedding match at or above the receipt's "
        "quarantine threshold",
        "flagged_groups": len(flagged),
        "sub_arms": report,
        "views": views_report,
    }
    admission["outputs"] = outputs
    _write_new(final / "admission.json", _json_bytes(admission))
    (args.out_run / "build").mkdir(mode=0o700)
    shutil.copy2(
        args.from_run / "build" / "build-manifest.json",
        args.out_run / "build" / "build-manifest.json",
    )
    screens = args.from_run / "screens"
    if screens.is_dir():
        (args.out_run / "screens").mkdir(mode=0o700)
        for path in sorted(screens.glob("*.json")):
            if path.name.endswith((".overlap.public.json", ".shortcut.json")):
                shutil.copy2(path, args.out_run / "screens" / path.name)
    print(
        json.dumps(
            {name: info["rows_out"] for name, info in report.items()}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
