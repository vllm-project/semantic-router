"""Make fresh Decision 2.0 SELECT/CAL from CSS pilot labels only.

The 15 CSS evaluation tasks are untouched. These holdouts are independent of
Decision 1.0 finetuning to the extent established by the legacy source audit,
but remain pilot monitoring data rather than release tests.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any

from transfer import build as transfer

from training.data import build_pilot as pilot
from training.model.data import check_partition_isolation, load_partition


def normalize_state(value: Any) -> str:
    return value if isinstance(value, str) else pilot.canonical(value)


def context_only(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "state": row["state"],
        "instructions": "",
        "options": [],
        "task_type": "context",
    }


def synthetic_type_row(seed: str, task_type: str, index: int) -> dict[str, Any]:
    family = f"css_pilot_independent_{task_type}"
    rng = pilot.rng_for(seed, family, index)
    if task_type == "noul":
        people = rng.sample(
            ("Mira", "Tomas", "Leena", "Omar", "Ravi", "Sana", "Yara"), 2
        )
        objects = rng.sample(
            ("blue folder", "bronze cup", "red scarf", "small atlas", "white lantern"),
            2,
        )
        place = rng.choice(("linen cabinet", "green locker", "oak drawer"))
        true_case = index % 2 == 0
        state = (
            f"The caretaker's note says that {people[0]} placed the {objects[0]} in the {place}. "
            f"Later, {people[1]} carried the {objects[1]} to the reading room. "
            "The note lists no further movement of either object."
        )
        instructions = f"At the end of the note, is the {objects[0] if true_case else objects[1]} in the {place}?"
        options = [
            {"key": key, "description": key.title()}
            for key in rng.sample(("true", "false"), 2)
        ]
        answer = "true" if true_case else "false"
        template = "caretaker_narrative_verification_v1"
        audit = {
            "oracle": answer,
            "queried_object": objects[0] if true_case else objects[1],
        }
    elif task_type == "score":
        title = rng.choice(("Orchid", "Maple", "Cedar", "Birch", "Juniper", "Willow"))
        score = index % 4
        present = set(
            rng.sample(("source citation", "clear date", "action summary"), score)
        )
        state = (
            "A writing judge gives one point for each of three features: a source citation, a clear date, "
            "and an action summary. The draft titled "
            + title
            + " has: "
            + "; ".join(
                f"{feature} {'present' if feature in present else 'absent'}"
                for feature in ("source citation", "clear date", "action summary")
            )
            + "."
        )
        options = [
            {"key": str(level), "description": f"{level} points"} for level in range(4)
        ]
        instructions = "What score from 0 to 3 does this one draft receive?"
        answer = str(score)
        template = "writing_judge_score_v1"
        audit = {"features_present": sorted(present), "oracle": score}
    else:
        raise ValueError(task_type)
    row = pilot._common_row(
        seed,
        family,
        index,
        state,
        instructions,
        options,
        answer,
        task_type,
        template,
        audit,
        "en",
    )
    row["source"] = "decision2_original_holdout_v1"
    row["audit_metadata"]["holdout_only"] = True
    return row


def build(args: argparse.Namespace) -> dict[str, Any]:
    panel_dir = args.panel_dir.resolve()
    legacy_path = args.legacy_train.resolve()
    output_dir = args.output_dir.resolve()
    panel_manifest_path = panel_dir / "css-manifest.json"
    panel_manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    if (
        panel_manifest["data_revision"] != transfer.DATA_REVISION
        or panel_manifest["replication_revision"] != transfer.REPLICATION_REVISION
    ):
        raise ValueError("CSS pilot panel source revisions differ from pinned protocol")
    files = {}
    for name in ("pilot_prompts", "pilot_gold"):
        info = panel_manifest["outputs"][name]
        path = panel_dir / info["file"]
        if pilot.sha_file(path) != info["sha256"]:
            raise ValueError(f"CSS {name} hash differs from panel manifest")
        files[name] = (path, info)
    prompts = [
        json.loads(line) for line in files["pilot_prompts"][0].open(encoding="utf-8")
    ]
    gold = [json.loads(line) for line in files["pilot_gold"][0].open(encoding="utf-8")]
    if len(prompts) != len(gold) or len(gold) != files["pilot_gold"][1]["n"]:
        raise ValueError("CSS pilot prompt/gold row counts differ")
    by_gold = {row["id"]: row for row in gold}
    if len(by_gold) != len(gold):
        raise ValueError("CSS pilot has duplicate IDs")
    rows = []
    for prompt in prompts:
        item = by_gold.pop(prompt["id"])
        if item["role"] != "pilot" or item["task"] not in transfer.PILOT_TASKS:
            raise ValueError("Non-pilot CSS gold encountered")
        question = prompt["questions"]["label"]
        if question["type"] != "choice":
            raise ValueError("CSS pilot question is not choice")
        criteria = question["criteria"]
        options = [
            {"key": str(key), "description": value} for key, value in criteria.items()
        ]
        answer = str(item["gold"])
        if answer not in criteria:
            raise ValueError(f"{prompt['id']}: pilot gold absent from criteria")
        state = prompt["state"]
        raw_hash = pilot.sha_bytes(state.encode("utf-8"))
        norm_hash = transfer.normalized_context_sha256(state)
        if (
            raw_hash != item["source_context_sha256"]
            or norm_hash != item["normalized_context_sha256"]
        ):
            raise ValueError(f"{prompt['id']}: panel context hash mismatch")
        row = {
            "id": prompt["id"],
            "state": state,
            "instructions": question["instructions"],
            "options": options,
            "label": next(
                i for i, option in enumerate(options) if option["key"] == answer
            ),
            "task_type": "choice",
            "family": f"css_pilot_{item['task']}",
            "group_id": f"css_pilot_context_{norm_hash}",
            "language": "en",
            "split": "select",
            "source": f"css_pilot:{item['task']}",
            "evaluation_role": "select",
            "render_template": "css_author_pilot_task_v1",
            "audit_metadata": {
                "task": item["task"],
                "source_id": item["source_id"],
                "source_context_sha256": raw_hash,
                "normalized_context_sha256": norm_hash,
            },
        }
        row["input_sha256"] = pilot.input_sha256(row)
        rows.append(row)
    if by_gold:
        raise ValueError("CSS pilot gold has unmatched prompt IDs")
    old_raw, old_norm = set(), set()
    old_rows = 0
    with legacy_path.open(encoding="utf-8") as stream:
        for line in stream:
            old_rows += 1
            legacy_row = json.loads(line)
            value = normalize_state(legacy_row["state"])
            old_raw.add(pilot.sha_bytes(value.encode("utf-8")))
            old_norm.add(transfer.normalized_context_sha256(value))
    raw_overlap = [
        row for row in rows if row["audit_metadata"]["source_context_sha256"] in old_raw
    ]
    norm_overlap = [
        row
        for row in rows
        if row["audit_metadata"]["normalized_context_sha256"] in old_norm
    ]
    excluded_groups = {row["group_id"] for row in [*raw_overlap, *norm_overlap]}
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    repeated_groups = {group for group, members in by_group.items() if len(members) > 1}
    excluded_groups.update(repeated_groups)
    excluded_detail = [
        {
            "group_id": group,
            "rows": len(by_group[group]),
            "reasons": sorted(
                (
                    ["legacy_raw_text"]
                    if any(x in raw_overlap for x in by_group[group])
                    else []
                )
                + (
                    ["legacy_normalized_text"]
                    if any(x in norm_overlap for x in by_group[group])
                    else []
                )
                + (["repeated_css_context"] if group in repeated_groups else [])
            ),
        }
        for group in sorted(excluded_groups)
    ]
    available = [row for row in rows if row["group_id"] not in excluded_groups]
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in available:
        by_task[row["audit_metadata"]["task"]].append(row)
    select, cal = [], []
    ordered_by_task = {}
    for task in transfer.PILOT_TASKS:
        candidates = sorted(
            by_task[task],
            key=lambda row: pilot.sha_bytes(
                f"{args.seed}\0css-pilot\0{row['id']}".encode()
            ),
        )
        if len(candidates) < args.select_per_task + args.cal_per_task:
            raise ValueError(f"{task}: too few independent pilot rows")
        select.extend(candidates[: args.select_per_task])
        ordered_by_task[task] = candidates[args.select_per_task :]
    possible_cal = [
        row for task in transfer.PILOT_TASKS for row in ordered_by_task[task]
    ]
    near_select = pilot.near_duplicates(
        [context_only(row) for row in possible_cal],
        [context_only(row) for row in select],
        collect_left_ids=True,
    )
    near_blocked_ids = set(near_select["left_ids"])
    for task in transfer.PILOT_TASKS:
        choices = [
            row for row in ordered_by_task[task] if row["id"] not in near_blocked_ids
        ]
        if len(choices) < args.cal_per_task:
            raise ValueError(f"{task}: too few context-distinct CAL candidates")
        cal.extend(choices[: args.cal_per_task])
    synthetic_select = []
    synthetic_cal = []
    for task_type in ("noul", "score"):
        generated = [
            synthetic_type_row(args.seed, task_type, index)
            for index in range(
                args.synthetic_select_per_type + args.synthetic_cal_per_type
            )
        ]
        synthetic_select.extend(generated[: args.synthetic_select_per_type])
        synthetic_cal.extend(generated[args.synthetic_select_per_type :])
    select.extend(synthetic_select)
    cal.extend(synthetic_cal)
    # Audit every selected source context against all Decision 1.0 finetuning
    # source contexts, including the new original holdout generators.
    for row in [*synthetic_select, *synthetic_cal]:
        value = normalize_state(row["state"])
        if (
            pilot.sha_bytes(value.encode("utf-8")) in old_raw
            or transfer.normalized_context_sha256(value) in old_norm
        ):
            raise ValueError(
                f"{row['id']}: synthetic holdout overlaps legacy source context"
            )
    for row in select:
        row["split"], row["evaluation_role"] = "select", "select"
    for row in cal:
        row["split"], row["evaluation_role"] = "cal", "calibrate"
    audit = pilot.overlap_audit(select, cal)
    # The shared CSS rubric dominates short text, causing many false positive
    # whole-prompt near matches. Compare source contexts separately instead.
    audit["near_duplicate"] = pilot.near_duplicates(
        [context_only(row) for row in select], [context_only(row) for row in cal]
    )
    if pilot.audit_has_exact_overlap(audit) or audit["near_duplicate"]["count"]:
        css_near = pilot.near_duplicates(
            [context_only(row) for row in select if row["id"].startswith("css/")],
            [context_only(row) for row in cal if row["id"].startswith("css/")],
        )
        synthetic_near = pilot.near_duplicates(
            [context_only(row) for row in synthetic_select],
            [context_only(row) for row in synthetic_cal],
        )
        raise ValueError(
            f"Fresh CSS SELECT/CAL share an exact or near input: "
            f"{audit['near_duplicate']['count']} near, CSS {css_near['count']}, synthetic {synthetic_near['count']}; "
            f"{audit['near_duplicate']['examples'][:5]}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    lengths = {
        partition: [pilot.count_tokens(row, tokenizer) for row in members]
        for partition, members in (("select", select), ("cal", cal))
    }
    if max(max(value) for value in lengths.values()) > args.max_row_tokens:
        raise ValueError("CSS pilot holdout exceeds max row tokens")
    output_files = {
        "select.jsonl": pilot.jsonl_bytes(select),
        "cal.jsonl": pilot.jsonl_bytes(cal),
    }
    if output_dir.exists():
        raise FileExistsError("Choose a fresh CSS holdout directory")
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, payload in output_files.items():
        pilot._atomic_write(output_dir / name, payload)
    check_partition_isolation(
        {
            "select": load_partition(output_dir / "select.jsonl", "select"),
            "cal": load_partition(output_dir / "cal.jsonl", "cal"),
        }
    )
    manifest = {
        "schema_version": "decision2-css-pilot-holdouts/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "legacy_source": {"sha256": pilot.sha_file(legacy_path), "rows": old_rows},
        "css_panel": {
            "manifest_sha256": pilot.sha_file(panel_manifest_path),
            "pilot_prompts_sha256": files["pilot_prompts"][1]["sha256"],
            "pilot_gold_sha256": files["pilot_gold"][1]["sha256"],
            "data_revision": transfer.DATA_REVISION,
            "replication_revision": transfer.REPLICATION_REVISION,
        },
        "source_context_overlap": {
            "legacy_raw_rows": len(raw_overlap),
            "legacy_normalized_rows": len(norm_overlap),
            "excluded_groups": excluded_detail,
            "excluded_group_ids_sha256": pilot.sha_bytes(
                pilot.canonical(sorted(excluded_groups)).encode()
            ),
        },
        "selection": {
            "seed_sha256": pilot.sha_bytes(args.seed.encode()),
            "select_per_task": args.select_per_task,
            "cal_per_task": args.cal_per_task,
            "synthetic_select_per_type": args.synthetic_select_per_type,
            "synthetic_cal_per_type": args.synthetic_cal_per_type,
            "cal_candidates_near_select_excluded": len(near_blocked_ids),
            "cal_candidates_near_select_ids_sha256": pilot.sha_bytes(
                pilot.canonical(sorted(near_blocked_ids)).encode()
            ),
            "unused_pilot_rows": len(available)
            - len(transfer.PILOT_TASKS) * (args.select_per_task + args.cal_per_task),
        },
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "rows": len(members),
                "task_counts": dict(
                    sorted(
                        collections.Counter(
                            row["audit_metadata"].get("task", row["family"])
                            for row in members
                        ).items()
                    )
                ),
                "task_type_counts": dict(
                    sorted(
                        collections.Counter(row["task_type"] for row in members).items()
                    )
                ),
                "token_total": sum(lengths[partition]),
                "token_maximum": max(lengths[partition]),
            }
            for partition, (name, payload, members) in {
                "select": ("select.jsonl", output_files["select.jsonl"], select),
                "cal": ("cal.jsonl", output_files["cal.jsonl"], cal),
            }.items()
        },
        "select_cal_audit": audit,
        "tokenizer_revision": args.tokenizer_revision,
        "rights_status": "Private evaluation only; SALT bundle has no top-level data license, original task rights vary.",
        "independence_scope": "No exact or normalized context overlap with the 24k Decision 1.0 train source; pilot tasks only. Public pretraining and semantic overlap are not ruled out.",
        "evaluation_policy": "May select checkpoints/calibrate; never use these pilot labels for training or final release claims.",
    }
    pilot._atomic_write(
        output_dir / "holdouts.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--legacy-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--select-per-task", type=int, default=200)
    parser.add_argument("--cal-per-task", type=int, default=100)
    parser.add_argument("--synthetic-select-per-type", type=int, default=0)
    parser.add_argument("--synthetic-cal-per-type", type=int, default=100)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args(argv)
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "select": manifest["outputs"]["select.jsonl"],
                "cal": manifest["outputs"]["cal.jsonl"],
                "legacy_context_overlap": manifest["source_context_overlap"],
            }
        )
    )


if __name__ == "__main__":
    main()
