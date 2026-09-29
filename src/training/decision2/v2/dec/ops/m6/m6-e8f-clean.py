"""Build `m6-e8f-r2clean` (decoder M6 prereg, arm E6K): E8F's TRAIN file minus
  (a) every row of the r2 rescreen's excluded groups (the overlap-effects payload: matched by group id,
      row id or input hash, as `v2.eval.overlap_effects exposure` matches),
  (b) A7 rows that A7 v3 removed relative to A7 v2 (E8F A7 ids absent from the A7 v3 arm files),
  (c) the A7-quarantined rows named by the DEV2.0-0.8B release (the release-support id list).
Rows are kept byte-identical and in order; a row matching several reasons is counted under the first.
Native token counts of the removed rows use the builder's `encode` lengths, so the output
manifest's tokens equal E8F's minus the removed rows.

usage (launch.sh --cpu): python3 v2/dec/ops/m6/m6-e8f-clean.py --train <E8F train> --expect-sha256 <sha>
    --excluded-groups <overlap payload json> --a7v3 <A7 v3 arm train file>... --quarantine <ids json>
    --tokenizer <path> --output <train.jsonl>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from training.model.data import file_sha256, load_partition  # noqa: E402
from v2.dec.build_mixture import token_lengths  # noqa: E402
from v2.dec.m5_block import component_slices  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--expect-sha256", required=True)
    p.add_argument("--excluded-groups", type=Path, required=True)
    p.add_argument("--a7v3", type=Path, action="append", required=True)
    p.add_argument("--a7-component", default="A7")
    p.add_argument("--quarantine", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sha = file_sha256(a.train)
    if sha != a.expect_sha256:
        raise ValueError(f"{a.train}: sha256 {sha} != {a.expect_sha256}")
    man = json.loads(a.train.with_name(a.train.name + ".manifest.json").read_text())
    lines = a.train.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    component = {}
    for name, items in component_slices(list(range(len(rows))), man):
        for i in items:
            component[i] = name
    payload = json.loads(a.excluded_groups.read_text())["groups"]
    excluded = set(payload)
    ex_ids = {i for g in payload.values() for i in g["row_ids"]}
    ex_hashes = {h for g in payload.values() for h in g["input_sha256"]}

    def exposed(r):
        return (
            r["group_id"] in excluded
            or r["id"] in ex_ids
            or r["input_sha256"] in ex_hashes
        )

    v3_ids = set()
    for path in a.a7v3:
        v3_ids.update(r["id"] for r in load_partition(path, "train"))
    quarantine = set(json.loads(a.quarantine.read_text()))
    reasons = {}
    for i, r in enumerate(rows):
        if exposed(r):
            reasons[i] = "a_r2_excluded_group"
        elif component[i] == a.a7_component and r["id"] not in v3_ids:
            reasons[i] = "b_a7v3_removed"
        elif r["id"] in quarantine:
            reasons[i] = "c_a7_quarantine"
    # Every reason's full membership, overlaps included (reported, not used for removal order)
    membership = {
        "a_r2_excluded_group": sum(exposed(r) for r in rows),
        "a_by_group_id_only": sum(r["group_id"] in excluded for r in rows),
        "b_a7v3_removed": sum(
            component[i] == a.a7_component and r["id"] not in v3_ids
            for i, r in enumerate(rows)
        ),
        "c_a7_quarantine": sum(r["id"] in quarantine for r in rows),
    }
    removed = [rows[i] for i in sorted(reasons)]
    lengths = dict(
        zip(
            (r["id"] for r in removed),
            token_lengths(removed, a.tokenizer, a.workers),
        )
    )
    pending = a.output.with_name(a.output.name + ".pending")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with pending.open("x", encoding="utf-8") as out:
        for i, line in enumerate(lines):
            if i not in reasons:
                out.write(line)
    pending.replace(a.output)
    kept = load_partition(a.output, "train")
    components = {}
    for name in [c["name"] for c in man["spec"]["components"]]:
        gone = [rows[i] for i in reasons if component[i] == name]
        components[name] = {
            "rows": man["components"][name]["rows"] - len(gone),
            "tokens": man["components"][name]["tokens"]
            - sum(lengths[r["id"]] for r in gone),
            "removed_rows": len(gone),
        }
    by_reason = Counter(reasons.values())
    manifest = {
        "schema_version": "dec-m6-e8f-clean/1",
        "source_train": str(a.train),
        "source_train_sha256": sha,
        "source_rows": len(rows),
        "source_tokens": man["tokens"],
        "excluded_groups_sha256": file_sha256(a.excluded_groups),
        "excluded_groups": len(excluded),
        "a7v3_files_sha256": {str(p): file_sha256(p) for p in a.a7v3},
        "quarantine_sha256": file_sha256(a.quarantine),
        "quarantine_ids": len(quarantine),
        "quarantine_ids_in_train": membership["c_a7_quarantine"],
        "removed_by_reason": dict(sorted(by_reason.items())),
        "membership_by_reason": membership,
        "removed_groups_by_reason": {
            k: len({rows[i]["group_id"] for i, v in reasons.items() if v == k})
            for k in sorted(by_reason)
        },
        "removed_by_component": dict(
            Counter(f"{component[i]}:{v}" for i, v in reasons.items())
        ),
        "removed_by_source": dict(
            Counter(f"{rows[i]['source']}:{v}" for i, v in reasons.items())
        ),
        "removed_tokens": sum(lengths.values()),
        "removed_ids": {
            k: sorted(rows[i]["id"] for i, v in reasons.items() if v == k)
            for k in sorted(by_reason)
        },
        "components": components,
        "component_order": [c["name"] for c in man["spec"]["components"]],
        "spec": man["spec"],
        "rows": len(kept),
        "tokens": man["tokens"] - sum(lengths.values()),
        "rows_by_type": dict(Counter(r["task_type"] for r in kept)),
        "rows_by_language": dict(Counter(r["language"] for r in kept).most_common()),
        "excluded_group_rows_left": sum(exposed(r) for r in kept),
        "output_sha256": file_sha256(a.output),
    }
    a.output.with_name(a.output.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("rows", "tokens", "removed_by_reason", "membership_by_reason")
            }
        )
    )


if __name__ == "__main__":
    main()
