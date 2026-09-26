"""Build a private FLUTE train-only pool and an optional 1k mixed pilot arm.

The SALT CSS test panels are read only as ID and hash exclusion lists. Gold
labels for panel IDs are never accessed when materializing candidate rows.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from transfer import build as transfer

from training.data import build_pilot as pilot
from training.model.data import check_partition_isolation, load_partition

SOURCE = "css_flute_official_train"
FAMILY = "css_flute_figurative_type"
RIGHTS = {
    "license": "AFL-3.0 (official ColumbiaNLP/FLUTE dataset card)",
    "attribution": "Saakyan et al., FLUTE: Figurative Language Understanding through Textual Explanations",
    "evidence": "https://huggingface.co/datasets/ColumbiaNLP/FLUTE/tree/5f4405119f9b862196018194b0c53fde25ee25a6; SALT mappings.py jsonl_download points to official train.jsonl",
    "rights_status": "official_dataset_license_documented_private_training_only",
}


def read_panel_exclusions(
    panel_dir: Path,
) -> tuple[dict[str, set[Any]], dict[str, Any]]:
    manifest_path = panel_dir / "css-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["data_revision"] != transfer.DATA_REVISION
        or manifest["replication_revision"] != transfer.REPLICATION_REVISION
    ):
        raise ValueError("CSS panel source revisions differ from the pinned protocol")
    excluded: dict[str, set[Any]] = {
        "task_source_ids": set(),
        "raw_context_sha256": set(),
        "normalized_context_sha256": set(),
        "panel_input_sha256": set(),
    }
    receipts = {}
    for role in ("pilot", "evaluation"):
        info = manifest["outputs"][f"{role}_gold"]
        path = panel_dir / info["file"]
        if pilot.sha_file(path) != info["sha256"]:
            raise ValueError(f"{role} hash exclusion file differs from panel manifest")
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                # Project immediately. Never retain the gold field or option labels.
                excluded["task_source_ids"].add((item["task"], str(item["source_id"])))
                excluded["raw_context_sha256"].add(item["source_context_sha256"])
                excluded["normalized_context_sha256"].add(
                    item["normalized_context_sha256"]
                )
                excluded["panel_input_sha256"].add(item["input_sha256"])
                count += 1
        if count != info["n"]:
            raise ValueError(
                f"{role} hash exclusion row count differs from panel manifest"
            )
        receipts[role] = {"file": info["file"], "sha256": info["sha256"], "rows": count}
    return excluded, {
        "manifest_sha256": pilot.sha_file(manifest_path),
        "source_revisions": {
            "SALT": transfer.DATA_REVISION,
            "replication": transfer.REPLICATION_REVISION,
        },
        "panels": receipts,
    }


def make_candidate_rows(
    source_data: dict[str, Any],
    mappings: tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]],
    excluded: dict[str, set[Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(source_data) != {"context", "labels", "prompts"}:
        raise ValueError("Unexpected SALT FLUTE classification JSON shape")
    contexts, labels, prompts = (
        source_data[key] for key in ("context", "labels", "prompts")
    )
    if (
        not all(isinstance(value, dict) for value in (contexts, labels, prompts))
        or set(contexts) != set(labels)
        or set(labels) != set(prompts)
    ):
        raise ValueError("SALT FLUTE source ID maps differ")
    excluded_counts: collections.Counter[str] = collections.Counter()
    by_context: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for source_id in sorted(contexts):
        source_id = str(source_id)
        if ("flute", source_id) in excluded["task_source_ids"]:
            excluded_counts["panel_source_id"] += 1
            continue
        context = contexts[source_id]
        if not isinstance(context, str):
            raise ValueError(f"{source_id}: context is not text")
        raw_sha = pilot.sha_bytes(context.encode("utf-8"))
        normalized_sha = transfer.normalized_context_sha256(context)
        if raw_sha in excluded["raw_context_sha256"]:
            excluded_counts["panel_raw_text_sha256"] += 1
            continue
        if normalized_sha in excluded["normalized_context_sha256"]:
            excluded_counts["panel_normalized_text_sha256"] += 1
            continue
        # Labels and source prompts are accessed only after all panel exclusions.
        instructions, criteria = transfer.criteria_for(
            "flute", prompts[source_id], *mappings
        )
        answer = str(labels[source_id])
        if answer not in criteria:
            raise ValueError(
                f"{source_id}: train label is absent from the authors' choice mapping"
            )
        options = [
            {"key": key, "description": description}
            for key, description in criteria.items()
        ]
        row = {
            "id": f"css_train/flute/{source_id}",
            "state": context,
            "instructions": instructions,
            "options": options,
            "label": next(
                index for index, option in enumerate(options) if option["key"] == answer
            ),
            "task_type": "choice",
            "family": FAMILY,
            "group_id": f"css_flute_context_{normalized_sha}",
            "language": "en",
            "split": "train",
            "source": SOURCE,
            "evaluation_role": "train",
            "render_template": "css_author_prompt_classification_v1",
            "audit_metadata": {
                "task": "flute",
                "source_id": source_id,
                "source_context_sha256": raw_sha,
                "normalized_context_sha256": normalized_sha,
                "upstream_partition": "FLUTE official train",
            },
        }
        row["input_sha256"] = pilot.input_sha256(row)
        if row["input_sha256"] in excluded["panel_input_sha256"]:
            excluded_counts["panel_input_sha256"] += 1
            continue
        pilot.validate_train_row(row)
        by_context[normalized_sha].append(row)
    rows = []
    for normalized_sha, variants in sorted(by_context.items()):
        gold = {
            (row["options"][row["label"]]["key"], pilot.input_sha256(row))
            for row in variants
        }
        if len({key for key, _ in gold}) != 1:
            excluded_counts["conflicting_duplicate_context_rows"] += len(variants)
            continue
        rows.append(min(variants, key=lambda row: row["id"]))
        excluded_counts["same_context_duplicate_rows"] += len(variants) - 1
    if not rows:
        raise ValueError("No FLUTE train candidates after panel exclusions")
    return rows, {
        "source_rows": len(contexts),
        "candidate_rows": len(rows),
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "class_counts": dict(
            sorted(
                collections.Counter(
                    row["options"][row["label"]]["key"] for row in rows
                ).items()
            )
        ),
        "normalized_contexts_unique": len({row["group_id"] for row in rows})
        == len(rows),
    }


def sample_class_stratified(
    rows: list[dict[str, Any]], count: int, seed: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ValueError("Invalid FLUTE sample count")
    by_class: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_class[row["options"][row["label"]]["key"]].append(row)
    quotas = {key: count * len(value) // len(rows) for key, value in by_class.items()}
    remainder = count - sum(quotas.values())
    ranked = sorted(
        by_class, key=lambda key: (-(count * len(by_class[key]) % len(rows)), key)
    )
    for key in ranked[:remainder]:
        quotas[key] += 1
    selected = []
    for key in sorted(by_class):
        candidates = sorted(
            by_class[key],
            key=lambda row: hashlib.sha256(
                f"{seed}\0flute\0{row['id']}".encode()
            ).hexdigest(),
        )
        selected.extend(candidates[: quotas[key]])
    if len(selected) != count:
        raise AssertionError("FLUTE class quota sample has wrong size")
    return selected, {
        "strategy": "class-stratified-hash-v1",
        "sample_rows": count,
        "class_quotas": dict(sorted(quotas.items())),
        "selected_id_sha256": pilot.sha_bytes(
            pilot.canonical(sorted(row["id"] for row in selected)).encode()
        ),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    replication_root = args.replication_root.resolve()
    panel_dir = args.panel_dir.resolve()
    control_dir = args.control_dir.resolve()
    output_dir = args.output_dir.resolve()
    if (
        transfer.git_head(data_root) != transfer.DATA_REVISION
        or transfer.git_head(replication_root) != transfer.REPLICATION_REVISION
    ):
        raise ValueError("CSS source revisions differ from the pinned protocol")
    transfer.require_clean_tracked_files(data_root)
    transfer.require_clean_tracked_files(replication_root)
    excluded, panel_receipt = read_panel_exclusions(panel_dir)
    source_path = data_root / "css_data/flute/flute-classification.json"
    mappings = transfer.source_maps(replication_root / "pilot_jev.py")
    pool, candidate_receipt = make_candidate_rows(
        json.loads(source_path.read_text(encoding="utf-8")), mappings, excluded
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    pool_lengths = [pilot.count_tokens(row, tokenizer) for row in pool]
    if max(pool_lengths) > args.max_row_tokens:
        raise ValueError(f"FLUTE train pool has row above {args.max_row_tokens} tokens")
    sample, sample_receipt = sample_class_stratified(pool, args.sample_count, args.seed)
    control_file = control_dir / "legacy_6k.train.jsonl"
    select_file = args.select_file.resolve()
    cal_file = args.cal_file.resolve()
    control = load_partition(control_file, "train")
    select = load_partition(select_file, "select")
    cal = load_partition(cal_file, "cal")
    if len(control) != 6000:
        raise ValueError("Control is not the frozen 6k baseline")
    length_cache = {
        row["id"]: pilot.count_tokens(row, tokenizer) for row in [*control, *sample]
    }
    keep, replacement = pilot.stratified_legacy_subset(
        control,
        len(control) - len(sample),
        sample,
        lambda row: length_cache[row["id"]],
        args.seed,
    )
    mixed = keep + sample
    pilot.rng_for(args.seed, "css-flute-mixed", 0).shuffle(mixed)
    check_partition_isolation({"train": mixed, "select": select, "cal": cal})
    checks = {
        "train_select": pilot.overlap_audit(mixed, select),
        "train_cal": pilot.overlap_audit(mixed, cal),
    }
    for name, result in checks.items():
        if pilot.audit_has_exact_overlap(result) or result["near_duplicate"]["count"]:
            raise ValueError(f"{name}: exact or near overlap")
    if pilot.train_consistency_audit(mixed)["conflicting_gold_groups"]:
        raise ValueError("Mixed arm contains conflicting gold for identical inputs")
    lengths = [length_cache[row["id"]] for row in mixed]
    if max(lengths) > args.max_row_tokens:
        raise ValueError("Mixed arm has overlength row")
    pool_payload = pilot.jsonl_bytes(pool)
    mixed_payload = pilot.jsonl_bytes(mixed)
    output_files = {
        "flute_pool.train.jsonl": pool_payload,
        "css_flute_1k.train.jsonl": mixed_payload,
        "legacy_6k.train.jsonl": control_file.read_bytes(),
        "select.jsonl": select_file.read_bytes(),
        "cal.jsonl": cal_file.read_bytes(),
    }
    if output_dir.exists():
        raise FileExistsError("Choose a fresh CSS train output directory")
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, payload in output_files.items():
        pilot._atomic_write(output_dir / name, payload)
    manifest = {
        "schema_version": "decision2-css-flute-train/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "source_file": {
            "path": "css_data/flute/flute-classification.json",
            "sha256": pilot.sha_file(source_path),
            "upstream_partition": "official FLUTE train",
        },
        "rights": RIGHTS,
        "panel_exclusions": panel_receipt,
        "candidate_audit": candidate_receipt,
        "sample": sample_receipt,
        "control_origin": {
            "path": str(control_file),
            "sha256": pilot.sha_file(control_file),
        },
        "replacement": replacement,
        "counts": {
            "control": {
                "family": dict(
                    sorted(
                        collections.Counter(row["family"] for row in control).items()
                    )
                ),
                "task_type": dict(
                    sorted(
                        collections.Counter(row["task_type"] for row in control).items()
                    )
                ),
                "language": dict(
                    sorted(
                        collections.Counter(row["language"] for row in control).items()
                    )
                ),
                "source": dict(
                    sorted(
                        collections.Counter(row["source"] for row in control).items()
                    )
                ),
            },
            "mixed": {
                "family": dict(
                    sorted(collections.Counter(row["family"] for row in mixed).items())
                ),
                "task_type": dict(
                    sorted(
                        collections.Counter(row["task_type"] for row in mixed).items()
                    )
                ),
                "language": dict(
                    sorted(
                        collections.Counter(row["language"] for row in mixed).items()
                    )
                ),
                "source": dict(
                    sorted(collections.Counter(row["source"] for row in mixed).items())
                ),
            },
        },
        "token_audit": {
            "tokenizer_revision": args.tokenizer_revision,
            "max_row_tokens": args.max_row_tokens,
            "pool_total": sum(pool_lengths),
            "pool_maximum": max(pool_lengths),
            "control_total": sum(length_cache[row["id"]] for row in control),
            "mixed_total": sum(lengths),
            "mixed_maximum": max(lengths),
        },
        "cross_partition_audit": checks,
        "test_text_overlap": {
            "raw_context": 0,
            "normalized_context": 0,
            "input_sha256": 0,
        },
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "bytes": len(payload),
                "rows": sum(1 for byte in payload.splitlines() if byte),
            }
            for name, payload in output_files.items()
        },
        "evaluation_interpretation": "FLUTE test is same-task supervised after training; remove FLUTE from zero-shot transfer aggregate.",
        "limitations": [
            "Official FLUTE train is the source of the SALT task subset; panel test source IDs and exact/normalized text hashes are excluded.",
            "This audit cannot prove semantic independence of distinct FLUTE premises or paraphrases.",
            "The 1k mixture changes task, label, language and token mix; not a single-factor causal intervention.",
        ],
    }
    pilot._atomic_write(
        output_dir / "css_flute_train.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--replication-root", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument(
        "--select-file",
        type=Path,
        required=True,
        help="Fresh independent CSS pilot SELECT, not legacy-derived monitoring data",
    )
    parser.add_argument(
        "--cal-file",
        type=Path,
        required=True,
        help="Fresh independent CSS pilot CAL, not legacy-derived monitoring data",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args(argv)
    if args.sample_count != 1000:
        raise ValueError("This fixed comparison requires --sample-count 1000")
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "candidate_rows": manifest["candidate_audit"]["candidate_rows"],
                "mixed_rows": manifest["outputs"]["css_flute_1k.train.jsonl"]["rows"],
                "mixed_sha256": manifest["outputs"]["css_flute_1k.train.jsonl"][
                    "sha256"
                ],
                "mixed_tokens": manifest["token_audit"]["mixed_total"],
            }
        )
    )


if __name__ == "__main__":
    main()
