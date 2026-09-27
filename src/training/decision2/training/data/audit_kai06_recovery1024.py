"""Gold-free overlap gate for the frozen Kai 0.6B recovery micro-arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.data.audit_release4b_overlap import audit_panel, read_jsonl
from training.data.build_kai06_recovery1024 import sha_file

EXPECTED = {
    "train": (1024, "58b94ac987b2fe134a392310a8d9d63f3d0e996a6016884f6dafadf3ae5449da"),
    "select": (700, "429285511fe5737dabfc799f01e7715155ad0f913994d5d10f5fdb627f6b0976"),
    "cal": (700, "3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a"),
    "typed_dev": (
        1600,
        "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
    ),
    "css_pilot": (
        1430,
        "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
    ),
    "typed_final": (
        1600,
        "e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd",
    ),
    "css_final": (
        6547,
        "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    ),
    "public231": (
        231,
        "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
    ),
}


def _native(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if any(
        not {"id", "component_id", "state_text", "source_id"} <= row.keys()
        for row in rows
    ):
        raise ValueError(f"Unexpected native schema: {path.name}")
    return (
        [
            {"id": row["id"], "state": row["state_text"], "source": row["source_id"]}
            for row in rows
        ],
        {row["component_id"] for row in rows},
    )


def run(paths: dict[str, Path], output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for name, path in paths.items():
        expected_n, expected_sha = EXPECTED[name]
        if sha_file(path) != expected_sha:
            raise ValueError(f"Frozen {name} bytes changed")
        if sum(1 for _ in path.open(encoding="utf-8")) != expected_n:
            raise ValueError(f"Frozen {name} cardinality changed")

    train, train_components = _native(paths["train"])
    select, select_components = _native(paths["select"])
    cal_rows = [json.loads(line) for line in paths["cal"].open(encoding="utf-8")]
    if any(not {"id", "state", "group_id", "source"} <= row.keys() for row in cal_rows):
        raise ValueError("Unexpected CAL schema")
    cal = [{"id": row["id"], "state": row["state"]} for row in cal_rows]
    panels = {"select": select, "cal": cal}
    panels.update(
        {
            name: read_jsonl(paths[name], prompt=True)
            for name in (
                "typed_dev",
                "css_pilot",
                "typed_final",
                "css_final",
                "public231",
            )
        }
    )
    audits = {name: audit_panel(train, rows) for name, rows in panels.items()}
    source_group_overlap = {
        "select_components": sorted(train_components & select_components),
        "cal_groups": sorted(train_components & {row["group_id"] for row in cal_rows}),
    }
    collision = any(
        count > 0
        for audit in audits.values()
        for count in audit["counts"].values()
        if isinstance(count, int)
    ) or any(source_group_overlap.values())
    report = {
        "schema": "decision2-kai06-recovery1024-overlap/1",
        "input_sha256": {name: EXPECTED[name][1] for name in paths},
        "rows": {name: EXPECTED[name][0] for name in paths},
        "panel_audits": audits,
        "source_group_overlap": source_group_overlap,
        "status": (
            "HOLD_overlap_review" if collision else "PASS_observable_overlap_only"
        ),
        "limits": "Approximate near search can miss paraphrases, semantic overlap, prior model exposure and task-family overlap; prompt-only audit reads no gold labels.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.chmod(0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run({name: getattr(args, name) for name in EXPECTED}, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "panel_counts": {
                    name: audit["counts"]
                    for name, audit in result["panel_audits"].items()
                },
                "source_group_overlap": {
                    name: len(ids)
                    for name, ids in result["source_group_overlap"].items()
                },
                "output_sha256": sha_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
