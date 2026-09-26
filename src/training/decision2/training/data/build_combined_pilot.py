"""Combine audited 5k legacy, 300 oracle skill, and 700 FLUTE train rows."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path

from transfer.build import normalized_context_sha256

from training.data import build_css_flute as css
from training.data import build_pilot as pilot
from training.model.data import check_partition_isolation, load_partition


def build(args: argparse.Namespace) -> dict:
    baseline = load_partition(args.legacy_control / "legacy_6k.train.jsonl", "train")
    program_arm = load_partition(args.program_arm / "mixed_6k.train.jsonl", "train")
    flute_arm = load_partition(args.flute_arm / "css_flute_1k.train.jsonl", "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    program = [row for row in program_arm if row["source"] == pilot.SOURCE]
    flute_1k = [row for row in flute_arm if row["source"] == css.SOURCE]
    if len(baseline) != 6000 or len(program) != 300 or len(flute_1k) != 1000:
        raise ValueError("Expected exact audited 6000/300/1000 source row counts")
    flute_700, flute_selection = css.sample_class_stratified(flute_1k, 700, args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True, trust_remote_code=False
    )
    lengths = {
        row["id"]: pilot.count_tokens(row, tokenizer)
        for row in [*baseline, *program, *flute_700]
    }
    legacy_5k, replacement = pilot.stratified_legacy_subset(
        baseline,
        5000,
        [*program, *flute_700],
        lambda row: lengths[row["id"]],
        args.legacy_selection_seed,
    )
    flute_only_5k = {row["id"] for row in flute_arm if row["source"] != css.SOURCE}
    if {row["id"] for row in legacy_5k} != flute_only_5k:
        raise ValueError("Combined 5k legacy core differs from FLUTE-only ablation")
    mixed = [*legacy_5k, *program, *flute_700]
    pilot.rng_for(args.seed, "combined-pilot", 0).shuffle(mixed)
    if len(mixed) != 6000 or len({row["id"] for row in mixed}) != 6000:
        raise ValueError("Combined arm has wrong size or repeated ID")
    check_partition_isolation({"train": mixed, "select": select, "cal": cal})
    checks = {
        "train_select": pilot.overlap_audit(mixed, select),
        "train_cal": pilot.overlap_audit(mixed, cal),
    }
    for name, audit in checks.items():
        if pilot.audit_has_exact_overlap(audit) or audit["near_duplicate"]["count"]:
            raise ValueError(f"{name}: exact or near train/holdout overlap")
    consistency = pilot.train_consistency_audit(mixed)
    if consistency["conflicting_gold_groups"]:
        raise ValueError("Combined arm has conflicting gold for repeated inputs")
    selected_lengths = [lengths[row["id"]] for row in mixed]
    if max(selected_lengths) > args.max_row_tokens:
        raise ValueError("Combined arm has overlength row")
    panel_exclusion, panel_receipt = css.read_panel_exclusions(args.panel_dir)
    test_text_overlap = {"raw_context": 0, "normalized_context": 0, "panel_input": 0}
    for row in mixed:
        state = (
            row["state"]
            if isinstance(row["state"], str)
            else pilot.canonical(row["state"])
        )
        test_text_overlap["raw_context"] += (
            pilot.sha_bytes(state.encode()) in panel_exclusion["raw_context_sha256"]
        )
        test_text_overlap["normalized_context"] += (
            normalized_context_sha256(state)
            in panel_exclusion["normalized_context_sha256"]
        )
        test_text_overlap["panel_input"] += (
            row["input_sha256"] in panel_exclusion["panel_input_sha256"]
        )
    if any(test_text_overlap.values()):
        raise ValueError(f"Combined arm overlaps CSS test hashes: {test_text_overlap}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Choose a fresh combined-arm directory")
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    files = {
        "combined_6k.train.jsonl": pilot.jsonl_bytes(mixed),
        "legacy_6k.train.jsonl": (
            args.legacy_control / "legacy_6k.train.jsonl"
        ).read_bytes(),
        "select.jsonl": args.select_file.read_bytes(),
        "cal.jsonl": args.cal_file.read_bytes(),
    }
    for name, payload in files.items():
        pilot._atomic_write(output_dir / name, payload)
    baseline_manifest = json.loads(
        (args.legacy_control / "legacy_6k.manifest.json").read_text()
    )
    source_rights = {**baseline_manifest["source_attribution"], css.SOURCE: css.RIGHTS}
    manifest = {
        "schema_version": "decision2-combined-pilot/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "source_inputs": {
            "legacy_control_sha256": pilot.sha_file(
                args.legacy_control / "legacy_6k.train.jsonl"
            ),
            "program_arm_sha256": pilot.sha_file(
                args.program_arm / "mixed_6k.train.jsonl"
            ),
            "flute_arm_sha256": pilot.sha_file(
                args.flute_arm / "css_flute_1k.train.jsonl"
            ),
            "select_sha256": pilot.sha_file(args.select_file),
            "cal_sha256": pilot.sha_file(args.cal_file),
        },
        "arm_counts": {"legacy": 5000, "programmatic": 300, "css_flute": 700},
        "flute_700_selection": flute_selection,
        "legacy_replacement": replacement,
        "shared_core_with_flute_only": 5000,
        "counts": {
            "family": dict(
                sorted(collections.Counter(row["family"] for row in mixed).items())
            ),
            "task_type": dict(
                sorted(collections.Counter(row["task_type"] for row in mixed).items())
            ),
            "language": dict(
                sorted(collections.Counter(row["language"] for row in mixed).items())
            ),
            "source": dict(
                sorted(collections.Counter(row["source"] for row in mixed).items())
            ),
        },
        "token_audit": {
            "method": "decoder-v2 segmented exact tokenizer",
            "tokenizer_revision": args.tokenizer_revision,
            "total": sum(selected_lengths),
            "minimum": min(selected_lengths),
            "maximum": max(selected_lengths),
            "max_row_tokens": args.max_row_tokens,
            "legacy_control_total": sum(lengths[row["id"]] for row in baseline),
        },
        "overlap_audit": checks,
        "within_train_audit": consistency,
        "css_test_hash_audit": {
            "panel_exclusion_files": panel_receipt,
            "overlap": test_text_overlap,
        },
        "source_attribution": source_rights,
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "bytes": len(payload),
                "rows": len(payload.splitlines()),
            }
            for name, payload in files.items()
        },
        "evaluation_interpretation": "CSS FLUTE is supervised same-task for this arm; remove it from zero-shot transfer aggregate.",
        "limitations": [
            "This arm changes row types, language balance and total tokens alongside data content.",
            "Near-duplicate search is approximate and does not prove semantic independence.",
        ],
    }
    pilot._atomic_write(
        output_dir / "combined_6k.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-control", type=Path, required=True)
    parser.add_argument("--program-arm", type=Path, required=True)
    parser.add_argument("--flute-arm", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument(
        "--legacy-selection-seed",
        required=True,
        help="Use the same legacy replacement seed as the FLUTE-only arm",
    )
    parser.add_argument("--seed", required=True)
    args = parser.parse_args(argv)
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "train_sha256": manifest["outputs"]["combined_6k.train.jsonl"][
                    "sha256"
                ],
                "tokens": manifest["token_audit"]["total"],
                "rows": manifest["outputs"]["combined_6k.train.jsonl"]["rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
