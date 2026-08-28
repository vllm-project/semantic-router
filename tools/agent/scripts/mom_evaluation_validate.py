#!/usr/bin/env python3
"""Validate MoM evaluation manifests and result bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_SCHEMA = REPO_ROOT / "config/evaluation/schema/mom-evaluation-manifest.schema.json"
RESULT_SCHEMA = REPO_ROOT / "config/evaluation/schema/mom-eval-result.schema.json"
CORE_SUITE = REPO_ROOT / "config/evaluation/mom-core-suite/v1/manifest.yaml"
PACK_REGISTRY = REPO_ROOT / "config/evaluation/packs/registry.yaml"
BASELINE_PROTOCOL = REPO_ROOT / "config/evaluation/baseline-protocol/v1.yaml"
SCORECARD_INDEX = REPO_ROOT / "config/evaluation/scorecards/index.yaml"
MOM_TAG = "mixture-of-models"
CONTRACT_MARKER = "vllm-sr/mom-evaluation/v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def validate_manifest_document(manifest: dict[str, Any], schema: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(manifest), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.path)
        errors.append(f"{path}: {location or '<root>'}: {error.message}")

    core_ref = REPO_ROOT / str(manifest.get("core_suite_ref", ""))
    if not core_ref.is_file():
        errors.append(f"{path}: core_suite_ref missing: {core_ref}")

    baseline_ref = manifest.get("baseline_protocol_ref")
    if baseline_ref:
        baseline_path = REPO_ROOT / str(baseline_ref)
        if not baseline_path.is_file():
            errors.append(f"{path}: baseline_protocol_ref missing: {baseline_path}")

    registry = load_yaml(PACK_REGISTRY)
    known_packs = set((registry.get("packs") or {}).keys())
    for entrypoint, entry in (manifest.get("entrypoints") or {}).items():
        for pack_id in entry.get("extension_packs") or []:
            if pack_id not in known_packs:
                errors.append(f"{path}: entrypoint {entrypoint}: unknown extension pack {pack_id}")
        if not entry.get("baselines"):
            errors.append(f"{path}: entrypoint {entrypoint}: requires at least one baseline")

    mom = manifest.get("mom") or {}
    recipe_version = str(mom.get("recipe_version") or "")
    recipe_id = str(mom.get("recipe_id") or "")
    is_published_catalog = "/built-in/" in str(path).replace("\\", "/")
    if recipe_id and recipe_version and SCORECARD_INDEX.is_file() and is_published_catalog:
        index = load_yaml(SCORECARD_INDEX)
        scorecards = (index.get("scorecards") or {}).get(recipe_id, {})
        for entrypoint in (manifest.get("entrypoints") or {}):
            entry_key = entrypoint.split("/")[-1]
            versions = scorecards.get(entrypoint) or scorecards.get(entry_key) or {}
            if recipe_version not in versions:
                errors.append(
                    f"{path}: missing published scorecard for {entrypoint} version {recipe_version}"
                )
    return errors


def validate_result_document(result: dict[str, Any], schema: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(result), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.path)
        errors.append(f"{path}: {location or '<root>'}: {error.message}")
    return errors


def find_mom_recipes() -> list[Path]:
    root = REPO_ROOT / "config/recipes"
    manifests: list[Path] = []
    for metadata_path in root.rglob("metadata.yaml"):
        metadata = load_yaml(metadata_path)
        tags = {str(tag).lower() for tag in metadata.get("tags") or []}
        if MOM_TAG in tags or "mixture-of-model" in tags:
            eval_path = metadata_path.parent / "mom-evaluation.yaml"
            manifests.append(eval_path)
    return sorted(manifests)


def validate_model_card_evaluation(readme_path: Path) -> list[str]:
    errors: list[str] = []
    if not readme_path.is_file():
        return [f"{readme_path}: Model Card missing"]
    text = readme_path.read_text(encoding="utf-8")
    if CONTRACT_MARKER not in text:
        errors.append(f"{readme_path}: Evaluation section must reference {CONTRACT_MARKER}")
    required_fragments = (
        "evaluation-scorecard.md",
        "baseline comparison",
        "Evaluation contract",
    )
    lowered = text.lower()
    for fragment in required_fragments:
        if fragment.lower() not in lowered:
            errors.append(f"{readme_path}: Evaluation section missing required fragment: {fragment}")
    scorecard_path = readme_path.parent / "evaluation-scorecard.md"
    if not scorecard_path.is_file():
        errors.append(f"{readme_path}: missing generated scorecard fragment evaluation-scorecard.md")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Validate one mom-evaluation.yaml manifest")
    parser.add_argument("--result", type=Path, help="Validate one mom_eval_result.json bundle")
    parser.add_argument("--all-mom-recipes", action="store_true", help="Validate all MoM recipe manifests")
    parser.add_argument("--check-model-cards", action="store_true", help="Validate MoM Model Card evaluation sections")
    args = parser.parse_args()

    manifest_schema = load_json(MANIFEST_SCHEMA)
    result_schema = load_json(RESULT_SCHEMA)
    errors: list[str] = []

    if args.manifest:
        path = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
        errors.extend(validate_manifest_document(load_yaml(path), manifest_schema, path))
    elif args.all_mom_recipes:
        for manifest_path in find_mom_recipes():
            if not manifest_path.is_file():
                errors.append(f"{manifest_path}: mom-evaluation.yaml required for MoM-tagged recipe")
                continue
            errors.extend(validate_manifest_document(load_yaml(manifest_path), manifest_schema, manifest_path))
            if args.check_model_cards and "/built-in/" in str(manifest_path).replace("\\", "/"):
                errors.extend(validate_model_card_evaluation(manifest_path.parent / "README.md"))
    elif args.result:
        path = args.result if args.result.is_absolute() else REPO_ROOT / args.result
        errors.extend(validate_result_document(load_json(path), result_schema, path))
    else:
        parser.error("specify --manifest, --result, or --all-mom-recipes")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("mom evaluation validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
