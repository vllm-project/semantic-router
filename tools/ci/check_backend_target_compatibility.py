#!/usr/bin/env python3
"""Verify the public backend-target matrix and its executable evidence."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

MATRIX_PATH = Path("website/docs/installation/backend-target-compatibility.md")
BEGIN_MARKER = "<!-- BEGIN BACKEND TARGET COMPATIBILITY MATRIX -->"
END_MARKER = "<!-- END BACKEND TARGET COMPATIBILITY MATRIX -->"

SURFACES = (
    "Canonical YAML",
    "Docker / CLI",
    "Helm",
    "Operator",
    "Dashboard",
    "Maintained recipes",
)
STATUSES = {"Supported", "Adapter", "Partial", "Not expressible"}
TARGET_FORMS = {
    "Direct `endpoint` as `host[:port]`",
    "HTTP(S) `base_url`, including a path",
    "Multiple weighted refs with shared route metadata",
    "Provider, API-version, path, and header metadata",
    "Kubernetes Service DNS target",
    "KServe discovery",
    "Label-selected Service discovery",
    "Different paths or request headers per weighted ref",
}

# Tests remain in their owning subsystems. This bridge makes removal or rename
# of that evidence fail together with the public compatibility claim.
EVIDENCE = {
    Path("config/config.yaml"): (
        "remote-secondary",
        "base_url: https://api.example.com/v1",
        "api_key_env: VLLM_SR_PRIMARY_API_KEY",
    ),
    Path("src/vllm-sr/tests/test_config_generator.py"): (
        "test_weighted_backend_refs_preserve_weights_and_shared_path",
        "test_backend_ref_https_base_url_uses_tls_and_explicit_extra_headers",
    ),
    Path("deploy/helm/testdata/backend-target-values.yaml"): (
        "base_url: https://provider.example/v1",
        "api_key_env: PROVIDER_API_KEY",
    ),
    Path("deploy/helm/validate-chart.sh"): (
        "Testing backend target compatibility rendering",
        "Canonical backend target fields were not preserved",
    ),
    Path("tools/make/helm.mk"): (
        "HELM_BACKEND_TARGET_VALUES",
        "Backend target compatibility rendering verified",
    ),
    Path("deploy/operator/controllers/semanticrouter_controller_test.go"): (
        "TestGenerateConfigYAMLIncludesBackendTargetAndLoRACatalog",
    ),
    Path("dashboard/frontend/src/pages/configPageModelFormSupport.test.ts"): (
        "preserves every canonical backend target field",
    ),
    Path("dashboard/backend/handlers/config_update_test.go"): (
        "TestUpdateConfigHandler_PreservesBackendTargetFields",
    ),
    Path("src/semantic-router/pkg/config/maintained_asset_contract_test.go"): (
        "TestMaintainedConfigAssetsUseCanonicalV03Contract",
    ),
}

MATRIX_ROW = re.compile(r"^\|\s*(?P<target>[^|]+?)\s*\|(?P<cells>.*)\|$")


def parse_matrix(matrix_path: Path) -> tuple[dict[str, list[str]], list[str]]:
    """Return target rows and structural errors from the marked public table."""
    text = matrix_path.read_text(encoding="utf-8")
    if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
        return {}, ["matrix must contain one begin marker and one end marker"]

    section = text.split(BEGIN_MARKER, 1)[1].split(END_MARKER, 1)[0]
    rows: dict[str, list[str]] = {}
    errors: list[str] = []
    header_seen = False

    for line in section.splitlines():
        match = MATRIX_ROW.match(line)
        if match is None:
            continue
        target = match.group("target").strip()
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        if target == "Target form":
            header_seen = True
            if cells != list(SURFACES):
                errors.append("matrix surface columns do not match the contract")
            continue
        if target.startswith("---"):
            continue
        if target in rows:
            errors.append(f"duplicate target form: {target}")
        rows[target] = cells

    if not header_seen:
        errors.append("matrix header is missing")
    return rows, errors


def validate(repo_root: Path, matrix_path: Path | None = None) -> list[str]:
    """Return public-matrix and evidence-drift errors."""
    matrix_path = matrix_path or repo_root / MATRIX_PATH
    if not matrix_path.is_file():
        return [f"backend target matrix is missing: {matrix_path}"]

    rows, errors = parse_matrix(matrix_path)
    actual_targets = set(rows)
    missing = TARGET_FORMS - actual_targets
    unexpected = actual_targets - TARGET_FORMS
    if missing:
        errors.append("missing target forms: " + ", ".join(sorted(missing)))
    if unexpected:
        errors.append("unexpected target forms: " + ", ".join(sorted(unexpected)))

    for target, cells in rows.items():
        if len(cells) != len(SURFACES):
            errors.append(
                f"{target}: expected {len(SURFACES)} surface cells, got {len(cells)}"
            )
            continue
        invalid = sorted(set(cells) - STATUSES)
        if invalid:
            errors.append(f"{target}: invalid statuses: {', '.join(invalid)}")

    for relative_path, markers in EVIDENCE.items():
        evidence_path = repo_root / relative_path
        if not evidence_path.is_file():
            errors.append(f"evidence file is missing: {relative_path}")
            continue
        evidence_text = evidence_path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in evidence_text:
                errors.append(f"evidence marker missing from {relative_path}: {marker}")

    return errors


def validate_canonical_config(config: object) -> list[str]:
    """Check the cross-reference contract needed by the Helm fixture."""
    if not isinstance(config, dict):
        return ["rendered config.yaml must be a mapping"]

    providers = config.get("providers")
    routing = config.get("routing")
    models = providers.get("models") if isinstance(providers, dict) else None
    cards = routing.get("modelCards") if isinstance(routing, dict) else None
    decisions = routing.get("decisions") if isinstance(routing, dict) else None
    if not isinstance(models, list) or not models:
        return ["rendered config.yaml must define providers.models"]
    if not isinstance(cards, list) or not cards:
        return ["rendered config.yaml must define routing.modelCards"]
    if not isinstance(decisions, list) or not decisions:
        return ["rendered config.yaml must define routing.decisions"]

    model_names = {model.get("name") for model in models if isinstance(model, dict)}
    card_names = {card.get("name") for card in cards if isinstance(card, dict)}
    errors: list[str] = []
    if missing_cards := sorted(name for name in model_names - card_names if name):
        errors.append(
            "rendered provider models lack routing.modelCards entries: "
            + ", ".join(missing_cards)
        )

    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"rendered routing.decisions[{index}] must be a mapping")
            continue
        if not isinstance(decision.get("priority"), int) or decision["priority"] <= 0:
            errors.append(
                f"rendered routing.decisions[{index}] must define positive priority"
            )
        refs = decision.get("modelRefs")
        if not isinstance(refs, list) or not refs:
            errors.append(f"rendered routing.decisions[{index}] must define modelRefs")
            continue
        for ref in refs:
            name = ref.get("model") if isinstance(ref, dict) else None
            if not name or name not in model_names or name not in card_names:
                errors.append(
                    f"rendered routing.decisions[{index}] has unknown model ref: {name!r}"
                )
    return errors


def validate_rendered_helm(path: Path) -> list[str]:
    """Extract and validate the backend-target ConfigMap produced by Helm."""
    try:
        documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as error:
        return [f"cannot parse rendered Helm output {path}: {error}"]

    candidates: list[str] = []
    for document in documents:
        if not isinstance(document, dict) or document.get("kind") != "ConfigMap":
            continue
        data = document.get("data")
        config_text = data.get("config.yaml") if isinstance(data, dict) else None
        if isinstance(config_text, str) and "provider.example/v1" in config_text:
            candidates.append(config_text)
    if len(candidates) != 1:
        return [
            "rendered Helm output must contain exactly one backend-target config.yaml"
        ]

    try:
        config = yaml.safe_load(candidates[0])
    except yaml.YAMLError as error:
        return [f"rendered config.yaml is invalid YAML: {error}"]
    return validate_canonical_config(config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument("--matrix", type=Path, help="override the matrix path")
    parser.add_argument(
        "--rendered-helm",
        type=Path,
        help="also validate the canonical config.yaml embedded in a Helm render",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    matrix_path = args.matrix.resolve() if args.matrix else None
    errors = validate(repo_root, matrix_path)
    if args.rendered_helm:
        errors.extend(validate_rendered_helm(args.rendered_helm.resolve()))
    if errors:
        for error in errors:
            print(f"backend target compatibility: {error}")
        return 1

    print(
        "backend target compatibility: "
        f"{len(TARGET_FORMS)} target forms and {len(EVIDENCE)} evidence contracts verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
