"""Add the audited short reasoning pool to the frozen human 9B train split.

Only already selected TRAIN rows are eligible. Protected inputs must contain
prompts without answers; the builder never reads evaluation labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

FROZEN = {
    "human_train": "e83fb07021b779bb86d6b1d773b007c2dda9d91052aedf1f72f89bebbfef50e2",
    "human_manifest": (
        "869a94c0c74b9e80f2b60bf414eb7440cda17cbce1e61906621bbe206ea5aa9f"
    ),
    "select": "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38",
    "cal": "bf5bbf29693928a2559ce0aff10e9d6b5b1541b50698634fcdb7725902412dcf",
    "short_train": "f2390fe0c5540c39d7ad8886fe9fc35054f253398092a841fb25bd411e894baa",
    "short_manifest": (
        "02cdf7e644e8f5fc75f6a908ceb5b6dcc6a7d7e172c047b0bcfc684d4e73be13"
    ),
}
PROTECTED = {
    "typed_final": "e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd",
    "css_final": "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    "typed_dev": "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
    "css_pilot": "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
    "public231": "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
}
SEED = "decision20-lux9b-short-replay-screen-20260927"


def selected_rows(
    pool: list[dict[str, Any]], selected_ids: set[str]
) -> list[dict[str, Any]]:
    """Require each source ID once and retain whole source groups only."""
    selected = [row for row in pool if row["id"] in selected_ids]
    if len(selected) != len(selected_ids) or len({r["id"] for r in selected}) != len(
        selected
    ):
        raise ValueError("Short-pool selected IDs are missing or duplicated")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        groups[row["group_id"]].append(row)
    for group in {row["group_id"] for row in selected}:
        if any(row["id"] not in selected_ids for row in groups[group]):
            raise ValueError("Selected short-pool group is incomplete")
    return selected


def quarantine(
    selected: list[dict[str, Any]], protected: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop a complete source group on any exact or approximate collision."""
    protected_ids = {row["id"] for row in protected}
    protected_groups = {row["group_id"] for row in protected if row.get("group_id")}
    protected_inputs = {
        row["input_sha256"] for row in protected if row.get("input_sha256")
    }
    exact_context = {targeted.text_hashes(row["state"]) for row in protected}
    raw = {item[0] for item in exact_context}
    normalized = {item[1] for item in exact_context}
    near = pilot.near_duplicates(
        targeted.context_rows(selected),
        targeted.context_rows(protected),
        collect_left_ids=True,
    )
    near_ids = set(near.pop("left_ids"))
    reject: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        group = row["group_id"]
        if (
            row["id"] in protected_ids
            or group in protected_groups
            or row["input_sha256"] in protected_inputs
        ):
            reject["exact_id_group_input"].add(group)
        state_hash = targeted.text_hashes(row["state"])
        if state_hash[0] in raw or state_hash[1] in normalized:
            reject["exact_or_normalized_context"].add(group)
        if row["id"] in near_ids:
            reject["approximate_near_context"].add(group)
    excluded = set().union(*reject.values()) if reject else set()
    kept = [row for row in selected if row["group_id"] not in excluded]
    return kept, {
        "excluded_group_counts_by_reason": {
            key: len(value) for key, value in sorted(reject.items())
        },
        "excluded_group_count": len(excluded),
        "approximate_near_hits": near["count"],
        "approximate_method": near["method"],
    }


def _prompt_only(
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if set(paths) != set(PROTECTED):
        raise ValueError("All five frozen answer-free panels are required")
    rows: list[dict[str, Any]] = []
    inventory = []
    for role, path in sorted(paths.items()):
        if "gold" in path.name.lower() or "label" in path.name.lower():
            raise ValueError("Only answer-free prompt files are permitted")
        if pilot.sha_file(path) != PROTECTED[role]:
            raise ValueError(f"Frozen {role} prompt digest mismatch")
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if set(row) != {"id", "state", "questions"}:
                    raise ValueError("Protected prompt contains unexpected keys")
                rows.append(
                    {
                        "id": f"{role}/{row['id']}",
                        "group_id": "",
                        "state": row["state"],
                    }
                )
                count += 1
        inventory.append({"role": role, "rows": count, "sha256": PROTECTED[role]})
    return rows, inventory


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inputs = {
        "human_train": args.human_train,
        "human_manifest": args.human_manifest,
        "select": args.select,
        "cal": args.cal,
        "short_train": args.short_train,
        "short_manifest": args.short_manifest,
    }
    for name, path in inputs.items():
        if pilot.sha_file(path) != FROZEN[name]:
            raise ValueError(f"Frozen {name} digest mismatch")
    human = load_partition(args.human_train, "train")
    select = load_partition(args.select, "select")
    cal = load_partition(args.cal, "cal")
    short = load_partition(args.short_train, "train")
    if (len(human), len(select), len(cal)) != (5824, 600, 900):
        raise ValueError("Frozen human partition counts changed")
    short_manifest = json.loads(args.short_manifest.read_text(encoding="utf-8"))
    if short_manifest.get("schema_version") != "decision20-short-reasoning-replay/1":
        raise ValueError("Unrecognized short-pool provenance")
    selected_ids = short_manifest.get("selected_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) != 270:
        raise ValueError("Audited short-pool selection changed")
    extra = selected_rows(short, set(selected_ids))
    human_ids = {item["id"] for item in human}
    if any(row["id"] in human_ids for row in extra):
        raise ValueError("Short pool overlaps human TRAIN IDs")
    prompt_rows, prompt_inventory = _prompt_only(
        {role: getattr(args, f"{role}_prompts") for role in PROTECTED}
    )
    extra, quarantine_report = quarantine(extra, [*human, *select, *cal, *prompt_rows])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = {row["id"]: pilot.count_tokens(row, tokenizer) for row in extra}
    too_long = {row["group_id"] for row in extra if lengths[row["id"]] > 1024}
    extra = [row for row in extra if row["group_id"] not in too_long]
    if not 180 <= len(extra) <= 270:
        raise ValueError(f"Insufficient audited short replay rows: {len(extra)}")
    human_tokens = sum(pilot.count_tokens(row, tokenizer) for row in human)
    extra_tokens = sum(lengths[row["id"]] for row in extra)
    share = extra_tokens / (human_tokens + extra_tokens)
    if share > 0.20:
        raise ValueError("Short replay exceeds 20% of 9B input tokens")
    merged = [*human, *extra]
    merged.sort(
        key=lambda row: (pilot.sha_bytes(f"{SEED}\0{row['id']}".encode()), row["id"])
    )
    check_partition_isolation({"train": merged, "select": select, "cal": cal})
    if pilot.train_consistency_audit(merged)["conflicting_gold_groups"]:
        raise ValueError("Merged TRAIN has contradictory gold labels")
    payload = pilot.jsonl_bytes(merged)
    args.output_dir.mkdir(mode=0o700, parents=True)
    pilot._atomic_write(args.output_dir / "train.jsonl", payload)
    manifest = {
        "schema_version": "decision20-lux9b-short-replay/1",
        "builder_sha256": pilot.sha_file(Path(__file__)),
        "seed": SEED,
        "input_sha256": FROZEN,
        "protected_prompts": prompt_inventory,
        "short_pool_selected": len(selected_ids),
        "short_pool_retained": len(extra),
        "retained_group_count": len({row["group_id"] for row in extra}),
        "quarantine": quarantine_report,
        "overlength_groups": len(too_long),
        "added_by_family": dict(
            sorted(Counter(row["family"] for row in extra).items())
        ),
        "tokens": {"human": human_tokens, "added": extra_tokens, "added_share": share},
        "outputs": {
            "train.jsonl": {"rows": len(merged), "sha256": pilot.sha_bytes(payload)}
        },
        "limits": [
            "Near-context matching cannot prove absence of paraphrases.",
            "Existing human-source task overlap remains documented separately.",
            "This builder never opens evaluation labels.",
        ],
    }
    pilot._atomic_write(
        args.output_dir / "manifest.json",
        (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "human-train",
        "human-manifest",
        "select",
        "cal",
        "short-train",
        "short-manifest",
        "tokenizer",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for role in PROTECTED:
        parser.add_argument(
            "--" + role.replace("_", "-") + "-prompts", type=Path, required=True
        )
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result["outputs"], sort_keys=True))


if __name__ == "__main__":
    main()
