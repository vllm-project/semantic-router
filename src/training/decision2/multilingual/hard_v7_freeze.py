"""Freeze r7 DEV commitments and make separated, gold-free locale handoffs.

Stage 1 contains only the local-language packet. Stage 2 holds the English
comparison packet and must be opened only after the reviewer seals Stage 1.
This tool neither scores nor performs model inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from multilingual.audit import sha256
from multilingual.hard_pilot_v7 import LANGUAGES, VERSION

EXPECTED_NATIVE_CORE_SHA256 = (
    "9ff8d754ce99c6539fc7f5bd88c10b196357c70124f59ed3d1bd1a0d3fdbb7d0"
)
FORBIDDEN_REVIEW_KEYS = {"gold", "answer", "semantic_gold", "facts", "operation"}


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def freeze(panel: Path, casebook: Path, source: Path, handoff: Path) -> dict:
    if handoff.exists() or (panel / "freeze.private.json").exists():
        raise FileExistsError("Frozen handoff/receipt already exists")
    manifest_path = panel / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != VERSION:
        raise ValueError("Unknown candidate version")
    if any(
        manifest.get(flag) is not False
        for flag in ("inference_eligible", "training_approved", "publication_eligible")
    ):
        raise ValueError("Candidate must remain blocked")
    if (
        sha256(source) != manifest["generator_sha256"]
        or sha256(casebook) != manifest["casebook_sha256"]
        or sha256(panel / "prompts.jsonl") != manifest["files_sha256"]["prompts.jsonl"]
        or sha256(panel / "targets.private.jsonl") != manifest["private_targets_sha256"]
    ):
        raise ValueError("Source/casebook/panel commitment mismatch")
    overlap = manifest["overlap_screen"]
    if (
        overlap["exact_matches"]
        or overlap["near_matches"]
        or overlap["reference_rows_with_state"] < 8855
    ):
        raise ValueError("Protected source overlap gate incomplete")
    native_path = panel / "native-preflight.json"
    native = json.loads(native_path.read_text(encoding="utf-8"))
    if (
        native.get("result") != "pass"
        or native.get("prompt_rows_checked") != 72
        or native.get("manifest_sha256") != sha256(manifest_path)
        or native.get("native_core_sha256") != EXPECTED_NATIVE_CORE_SHA256
    ):
        raise ValueError("Pinned native parser gate failed")
    reviewed = {}
    for language in LANGUAGES:
        name = f"review.gold-free.{language}.jsonl"
        path = panel / name
        if sha256(path) != manifest["files_sha256"][name]:
            raise ValueError(f"{language}: review packet commitment mismatch")
        rows = read_rows(path)
        if len(rows) != 18 or len({row["base_id"] for row in rows}) != 18:
            raise ValueError(f"{language}: incomplete independent bases")
        if any(
            row.get("language") != language
            or FORBIDDEN_REVIEW_KEYS & set(row)
            or FORBIDDEN_REVIEW_KEYS & set(row["question"])
            for row in rows
        ):
            raise ValueError(f"{language}: key or wrong locale in review packet")
        reviewed[language] = rows
    bases = {row["base_id"] for row in reviewed["en"]}
    if any({row["base_id"] for row in reviewed[lang]} != bases for lang in LANGUAGES):
        raise ValueError("Locale packets do not share the same 18 base cases")

    handoff.mkdir(parents=True, mode=0o700)
    handoff_hashes = {}
    for language in LANGUAGES:
        local_dir = handoff / language
        local_dir.mkdir(mode=0o700)
        stage1 = local_dir / "stage1-local.jsonl"
        shutil.copyfile(panel / f"review.gold-free.{language}.jsonl", stage1)
        stage1.chmod(0o400)
        files = {"stage1-local.jsonl": sha256(stage1)}
        if language != "en":
            stage2 = local_dir / "stage2-en-reference.jsonl"
            shutil.copyfile(panel / "review.gold-free.en.jsonl", stage2)
            stage2.chmod(0o400)
            files["stage2-en-reference.jsonl"] = sha256(stage2)
        guide = {
            "schema_version": "decision2-multilingual-hard-v7-review-handoff/1",
            "language": language,
            "qualification": "Independent qualified native or bilingual reviewer; non-native author self-review is insufficient",
            "stage1": "Solve each local-language row without English or any answer key; record answer, cited local rule/evidence, ambiguity and naturalness; seal all 18 judgments before Stage 2.",
            "stage2": (
                "After Stage 1 seal, compare with paired English text and record translation fidelity, missing conditions and conflicts; seal this assessment separately."
                if language != "en"
                else "No bilingual comparison stage for English."
            ),
            "prohibited": "No casebook, targets, oracle output or model predictions are supplied. Do not infer release eligibility from this packet.",
            "file_sha256": files,
        }
        guide_path = local_dir / "review-guide.json"
        guide_path.write_text(
            json.dumps(guide, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        guide_path.chmod(0o400)
        files["review-guide.json"] = sha256(guide_path)
        handoff_hashes[language] = files
    receipt = {
        "schema_version": "decision2-multilingual-hard-v7-freeze/1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": manifest["source_revision"],
        "generator_sha256": manifest["generator_sha256"],
        "freeze_source_sha256": sha256(Path(__file__)),
        "casebook_sha256": manifest["casebook_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "prompts_sha256": manifest["files_sha256"]["prompts.jsonl"],
        "targets_sha256": manifest["private_targets_sha256"],
        "native_preflight_sha256": sha256(native_path),
        "native_core_sha256": native["native_core_sha256"],
        "overlap": overlap,
        "handoff_file_sha256": handoff_hashes,
        "independence_unit": "18 base IDs; four language variants paired",
        "status": "BLOCK_FOR_INFERENCE_PENDING_QUALIFIED_BILINGUAL_REVIEW",
        "model_inference_count": 0,
        "training_approved": False,
        "publication_eligible": False,
    }
    receipt_path = panel / "freeze.private.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o400)
    for name in ("prompts.jsonl", "targets.private.jsonl", "manifest.json"):
        (panel / name).chmod(0o400)
    for language in LANGUAGES:
        (panel / f"review.gold-free.{language}.jsonl").chmod(0o400)
    return {
        "freeze_receipt_sha256": sha256(receipt_path),
        "manifest_sha256": receipt["manifest_sha256"],
        "status": receipt["status"],
        "locale_packets": {
            language: handoff_hashes[language]["stage1-local.jsonl"]
            for language in LANGUAGES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--casebook", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.panel, args.casebook, args.source, args.handoff), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
