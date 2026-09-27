"""Explicit post-key v3 package path; never rewrites the pre-key freeze.

The original package validator still checks model files, panel identities,
rights, provenance, native parity and reviewed evidence. This additive entry
point binds a new user-directed aggregate-first gate and appends a visible
post-key disclosure. A standalone verify command checks the published bytes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from publication import bundle_arena_v3 as bundle

PACKAGE_VERSION = "decision2-jevarena-v3-postkey-release-bundle/1"
GATE_VERSION = "decision2-jevarena-v3-postkey-release-gate/1"
GATE_STATUS = "passed_postkey_user_directed"
CARD_MARKER = "**Post-key, user-directed aggregate-priority release.**"
AMENDMENT = (
    Path(__file__).resolve().parents[1]
    / "research/jev-arena-v3-postkey-answer-count-erratum-2026-09-27.md"
)
EXTRA = (
    "postkey-amendment.md",
    "postkey-rank-diagnostic.json",
    "original-strict-hold.json",
)


def _frozen_ranker(source_root: Path) -> Any:
    source = source_root.resolve(strict=True) / "jev_arena/arena_v3.py"
    if source.is_symlink() or not source.is_file():
        raise ValueError("Original frozen ranker source is missing or linked")
    spec = importlib.util.spec_from_file_location("decision2_frozen_v3", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load original frozen ranker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != source:
        raise ValueError("Loaded ranker is not the original frozen file")
    return module


def _gate(
    gate: dict[str, Any],
    *,
    record_sha: str,
    parity_sha: str,
    artifact_sha: str,
    arena_sha: str,
    public_sha: str,
    freeze_sha: str,
    native_sha: str,
    model_files_sha: str,
    model_id: str,
    revision: str,
) -> None:
    required = {
        "package_record_sha256": record_sha,
        "parity_receipt_sha256": parity_sha,
        "artifact_manifest_sha256": artifact_sha,
        "arena_rank_sha256": arena_sha,
        "jevbench_public_rank_sha256": public_sha,
        "pretest_freeze_sha256": freeze_sha,
        "native_model_sha256": native_sha,
        "model_files_sha256": model_files_sha,
    }
    if (
        gate.get("schema_version") != GATE_VERSION
        or gate.get("status") != GATE_STATUS
        or gate.get("model_id") != model_id
        or gate.get("model_revision") != revision
        or any(gate.get(name) != value for name, value in required.items())
        or gate.get("amendment_sha256") != bundle.common.sha_file(AMENDMENT)
        or gate.get("minimum_aggregate_delta") != 3.0
        or gate.get("typed_answer_denominator") != 2000
        or gate.get("original_strict_gate") != "HOLD"
    ):
        raise ValueError("Post-key gate or frozen package binding is incomplete")
    for name in (
        "original_rank_failure_log_sha256",
        "original_strict_hold_sha256",
        "rank_diagnostic_sha256",
    ):
        bundle.common._sha(gate.get(name), name)
    checks = gate.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(bundle.GATE_CHECKS):
        raise ValueError("Post-key gate omits a required independent review")
    for name, check in checks.items():
        if not isinstance(check, dict) or check.get("status") != "passed":
            raise ValueError(f"Post-key review is blocked: {name}")
        bundle.common._sha(check.get("evidence_sha256"), f"{name} evidence")


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _config_paths(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    def path(name: str) -> Path:
        return _resolve(config_path, config[name])

    def mapping(name: str) -> dict[str, Path]:
        value = config.get(name)
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a receipt-to-path mapping")
        return {key: _resolve(config_path, item) for key, item in value.items()}

    return {
        "model_dir": path("model_dir"),
        "artifacts": path("artifacts"),
        "arena_rank": path("arena_rank"),
        "public_rank": path("public_rank"),
        "package_record": path("package_record"),
        "parity_receipt": path("parity_receipt"),
        "release_gate": path("release_gate"),
        "provenance_inputs": mapping("provenance_inputs"),
        "freeze_manifest": path("freeze_manifest"),
        "gate_evidence": mapping("gate_evidence"),
        "score_inputs": {
            family: {key: _resolve(config_path, value) for key, value in files.items()}
            for family, files in config["score_inputs"].items()
        },
        "score_key": config["score_key"],
        "aggregate_pair_report": path("aggregate_pair_report"),
        "old_typed_report": path("old_typed_report"),
        "old_css_report": path("old_css_report"),
        "base_source": path("base_source") if "base_source" in config else None,
        "adapter_source_parity_receipt": (
            path("adapter_source_parity_receipt")
            if "adapter_source_parity_receipt" in config
            else None
        ),
    }


@contextmanager
def _amended_bundle(
    frozen: Any, gate: dict[str, Any], strict_hold: dict[str, Any] | None = None
) -> Iterator[None]:
    old = {
        name: getattr(bundle, name)
        for name in (
            "checked_freeze",
            "SCORERS",
            "POLICY",
            "VERSION",
            "GATE_VERSION",
            "_numeric_release_gate",
            "_gate",
            "_card",
        )
    }
    strict_numeric = bundle._numeric_release_gate
    original_card = bundle._card
    original_public_text = bundle.common._public_text
    original_inventory = bundle.common._inventory
    original_model_suffixes = bundle.common.MODEL_SUFFIXES
    original_text_suffixes = bundle.common.TEXT_SUFFIXES

    def reviewed_public_text(content: str, label: str) -> None:
        # The unchanged native serve.py uses only loopback in examples/defaults.
        # Screen the original text for secrets and paths, and retain every other
        # address in the normal scanner. This exception cannot change file bytes.
        if label not in {"serve.py", "serve_vllm.sh"}:
            original_public_text(content, label)
            return
        masked = bundle.common.IP_ADDRESS.sub(
            lambda match: (
                "__REVIEWED_IPV4_LOOPBACK__"
                if match.group() == "127.0.0.1"
                else match.group()
            ),
            content,
        )
        original_public_text(masked, label)

    def reviewed_inventory(
        root: Path, *, allow_adapter_metadata: bool = False
    ) -> dict[str, str]:
        for path in root.rglob("*.sh"):
            if path.relative_to(root).as_posix() != "serve_vllm.sh":
                raise ValueError("Only the reviewed native serve_vllm.sh is supported")
        bundle.common.MODEL_SUFFIXES = original_model_suffixes | {".sh"}
        bundle.common.TEXT_SUFFIXES = original_text_suffixes | {".sh"}
        try:
            return original_inventory(
                root, allow_adapter_metadata=allow_adapter_metadata
            )
        finally:
            bundle.common.MODEL_SUFFIXES = original_model_suffixes
            bundle.common.TEXT_SUFFIXES = original_text_suffixes

    def amended_numeric(**kwargs: Any) -> dict[str, str]:
        try:
            strict_numeric(**kwargs)
        except ValueError as error:
            if "score slice regression exceeded 0.02" not in str(error):
                raise ValueError(
                    "Original strict gate did not reproduce the recorded Score HOLD"
                ) from error
        else:
            raise ValueError("Original strict Score HOLD did not reproduce")
        return strict_numeric(**kwargs, postkey_aggregate_priority=True)

    def amended_card(*args: Any, **kwargs: Any) -> str:
        card = original_card(*args, **kwargs)
        model_id = args[0]
        marker = f"# {model_id}\n"
        if card.count(marker) != 1:
            raise ValueError("Post-key disclosure has no unique card heading")
        score_tradeoff = ""
        if strict_hold is not None:
            candidate = strict_hold["candidate_score_accuracy_all"]
            comparator = strict_hold["comparator_score_accuracy_all"]
            score_tradeoff = (
                f"The Score slice fell from {100 * comparator:.2f}% to "
                f"{100 * candidate:.2f}% ({100 * (candidate - comparator):+.2f} "
                "percentage points), breaching the original 2.0-point floor. "
            )
        note = (
            f"\n{CARD_MARKER} The original predeclared Score-slice gate failed; "
            "the owner changed the release preference after FINAL labels were "
            "visible. This package uses the separately documented aggregate "
            "gain of at least +3.0 points with a positive paired 95% interval. "
            f"{score_tradeoff}Per-axis, per-type and per-task results remain disclosed. "
            f"Amendment SHA-256: `{gate['amendment_sha256']}`.\n"
        )
        return card.replace(marker, marker + note, 1)

    bundle.checked_freeze = frozen._freeze
    bundle.SCORERS = frozen.SCORER_SOURCE_PATHS
    bundle.POLICY = Path(frozen.__file__).resolve().parents[1] / (
        "research/jev-arena-v3-first-release-gates-2026-09-27.md"
    )
    bundle.VERSION = PACKAGE_VERSION
    bundle.GATE_VERSION = GATE_VERSION
    bundle._numeric_release_gate = amended_numeric
    bundle._gate = _gate
    bundle._card = amended_card
    bundle.common._public_text = reviewed_public_text
    bundle.common._inventory = reviewed_inventory
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(bundle, name, value)
        bundle.common._public_text = original_public_text
        bundle.common._inventory = original_inventory
        bundle.common.MODEL_SUFFIXES = original_model_suffixes
        bundle.common.TEXT_SUFFIXES = original_text_suffixes


def _validate_addenda(
    config: dict[str, Any],
    paths: dict[str, Any],
    diagnostic: Path,
    rank_failure_log: Path,
    strict_hold_path: Path,
) -> dict[str, Any]:
    gate = bundle.common._object(paths["release_gate"])
    if any(
        path.is_symlink() for path in (diagnostic, rank_failure_log, strict_hold_path)
    ):
        raise ValueError("Post-key evidence symlinks are not allowed")
    sidecar = bundle.common._object(diagnostic)
    if (
        sidecar.get("schema_version") != "jevarena-v3-postkey-answer-count-diagnostic/1"
        or sidecar.get("status") != "postkey_diagnostic_not_preregistered_release_gate"
        or sidecar.get("ranked_result") != bundle.common._object(paths["arena_rank"])
        or gate.get("rank_diagnostic_sha256") != bundle.common.sha_file(diagnostic)
        or gate.get("original_rank_failure_log_sha256")
        != bundle.common.sha_file(rank_failure_log)
        or sidecar.get("original_failed_rank_log_sha256")
        != gate["original_rank_failure_log_sha256"]
        or sidecar.get("prekey_freeze_sha256")
        != bundle.common.sha_file(paths["freeze_manifest"])
    ):
        raise ValueError("Post-key rank and original failure are not bound")
    strict_hold = bundle.common._object(strict_hold_path)
    new_typed_path = paths["score_inputs"]["typed"]["score"]
    old_typed_path = paths["old_typed_report"]
    new_typed = bundle.common._object(new_typed_path)
    old_typed = bundle.common._object(old_typed_path)
    new_score = new_typed.get("by_type", {}).get("score", {}).get("accuracy_all")
    old_score = old_typed.get("by_type", {}).get("score", {}).get("accuracy_all")
    if (
        type(new_score) not in (int, float)
        or type(old_score) not in (int, float)
        or not new_score < old_score - 0.02 - 1e-12
        or gate.get("original_strict_hold_sha256")
        != bundle.common.sha_file(strict_hold_path)
        or strict_hold
        != {
            "schema_version": "decision2-v3-strict-release-hold/1",
            "status": "HOLD",
            "reason": "typed_score_slice_regression",
            "prekey_freeze_sha256": bundle.common.sha_file(paths["freeze_manifest"]),
            "candidate_model_id": new_typed["model"]["id"],
            "comparator_model_id": old_typed["model"]["id"],
            "candidate_typed_report_sha256": bundle.common.sha_file(new_typed_path),
            "comparator_typed_report_sha256": bundle.common.sha_file(old_typed_path),
            "candidate_score_accuracy_all": new_score,
            "comparator_score_accuracy_all": old_score,
            "predeclared_max_absolute_regression": 0.02,
        }
    ):
        raise ValueError("Original strict Score HOLD receipt is missing or unbound")
    threshold_path = paths["gate_evidence"]["release_thresholds"]
    threshold = bundle.common._object(threshold_path)
    if (
        threshold.get("amendment_sha256") != gate.get("amendment_sha256")
        or threshold.get("rank_diagnostic_sha256") != gate.get("rank_diagnostic_sha256")
        or threshold.get("original_rank_failure_log_sha256")
        != gate.get("original_rank_failure_log_sha256")
        or threshold.get("original_strict_hold_sha256")
        != gate.get("original_strict_hold_sha256")
        or threshold.get("policy_phase") != "postkey_user_directed"
    ):
        raise ValueError("Release-threshold reviewer omitted the post-key amendment")
    if set(paths["gate_evidence"]) != set(bundle.GATE_CHECKS):
        raise ValueError("All six independent release checks are required")
    if set(paths["score_inputs"]) != set(bundle.SCORE_FAMILIES):
        raise ValueError("Typed, CSS and public native scores are required")
    if config.get("postkey_policy") != "aggregate_priority_delta_3":
        raise ValueError("Package config must identify the amended release policy")
    return gate


def _extra_files(package: Path, diagnostic: Path, strict_hold: Path) -> None:
    for source, name in (
        (AMENDMENT, EXTRA[0]),
        (diagnostic, EXTRA[1]),
        (strict_hold, EXTRA[2]),
    ):
        bundle.common._public_text(source.read_text(encoding="utf-8"), name)
        target = package / name
        shutil.copyfile(source, target)
    manifest_path = package / "PACKAGE_MANIFEST.json"
    manifest = bundle.common._object(manifest_path)
    inventory = manifest.get("files_sha256")
    if not isinstance(inventory, dict):
        raise TypeError("Incomplete staged package inventory")
    for name in EXTRA:
        inventory[name] = bundle.common.sha_file(package / name)
    manifest["postkey_amendment_sha256"] = inventory[EXTRA[0]]
    manifest["postkey_rank_diagnostic_sha256"] = inventory[EXTRA[1]]
    manifest["original_strict_hold_sha256"] = inventory[EXTRA[2]]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify(package: Path, frozen: Any) -> dict[str, Any]:
    gate = bundle.common._object(package / "release-gate.json")
    manifest = bundle.common._object(package / "PACKAGE_MANIFEST.json")
    for name in EXTRA:
        if (package / name).is_symlink() or not (package / name).is_file():
            raise ValueError(f"Missing post-key package disclosure: {name}")
    if (
        manifest.get("bundle_version") != PACKAGE_VERSION
        or manifest.get("postkey_amendment_sha256")
        != bundle.common.sha_file(package / EXTRA[0])
        or manifest.get("postkey_rank_diagnostic_sha256")
        != bundle.common.sha_file(package / EXTRA[1])
        or manifest["postkey_amendment_sha256"] != gate.get("amendment_sha256")
        or manifest["postkey_rank_diagnostic_sha256"]
        != gate.get("rank_diagnostic_sha256")
        or manifest.get("original_strict_hold_sha256")
        != bundle.common.sha_file(package / EXTRA[2])
        or manifest["original_strict_hold_sha256"]
        != gate.get("original_strict_hold_sha256")
        or CARD_MARKER not in (package / "README.md").read_text(encoding="utf-8")
    ):
        raise ValueError("Post-key package disclosure or digest differs")
    sidecar = bundle.common._object(package / EXTRA[1])
    if (
        sidecar.get("schema_version") != "jevarena-v3-postkey-answer-count-diagnostic/1"
        or sidecar.get("prekey_freeze_sha256") != gate.get("pretest_freeze_sha256")
        or sidecar.get("original_failed_rank_log_sha256")
        != gate.get("original_rank_failure_log_sha256")
    ):
        raise ValueError("Post-key package lacks the original freeze/failure binding")
    strict_hold = bundle.common._object(package / EXTRA[2])
    if (
        strict_hold.get("schema_version") != "decision2-v3-strict-release-hold/1"
        or strict_hold.get("status") != "HOLD"
        or strict_hold.get("reason") != "typed_score_slice_regression"
        or strict_hold.get("prekey_freeze_sha256") != gate.get("pretest_freeze_sha256")
    ):
        raise ValueError("Post-key package erased the original strict HOLD")
    with _amended_bundle(frozen, gate):
        result = bundle.verify(package)
    return result


def assemble(
    config_path: Path,
    output: Path,
    source_root: Path,
    diagnostic: Path,
    rank_failure_log: Path,
    strict_hold: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    config = bundle.common._object(config_path)
    paths = _config_paths(config_path, config)
    gate = _validate_addenda(config, paths, diagnostic, rank_failure_log, strict_hold)
    frozen = _frozen_ranker(source_root)
    freeze = bundle.common._object(paths["freeze_manifest"])
    if (
        freeze.get("score_sources_sha256", {}).get("arena_v3")
        != bundle.common.sha_file(Path(frozen.__file__))
        or gate.get("amendment_sha256") != bundle.common.sha_file(AMENDMENT)
        or gate.get("original_strict_gate") != "HOLD"
    ):
        raise ValueError("Original freeze and post-key amendment are unbound")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.postkey-{uuid.uuid4().hex}")
    with _amended_bundle(frozen, gate, bundle.common._object(strict_hold)):
        bundle.assemble(output=stage, **paths)
    _extra_files(stage, diagnostic, strict_hold)
    result = verify(stage, frozen)
    stage.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source-root", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--postkey-rank-diagnostic", type=Path)
    parser.add_argument("--original-rank-failure-log", type=Path)
    parser.add_argument("--original-strict-hold", type=Path)
    args = parser.parse_args()
    frozen = _frozen_ranker(args.frozen_source_root)
    if args.verify is not None:
        result = verify(args.verify, frozen)
    else:
        if any(
            value is None
            for value in (
                args.config,
                args.output,
                args.postkey_rank_diagnostic,
                args.original_rank_failure_log,
                args.original_strict_hold,
            )
        ):
            parser.error(
                "assembly requires config, output, rank diagnostic, rank failure log and strict HOLD receipt"
            )
        result = assemble(
            args.config,
            args.output,
            args.frozen_source_root,
            args.postkey_rank_diagnostic,
            args.original_rank_failure_log,
            args.original_strict_hold,
        )
    print(json.dumps({"model_id": result["model_id"], "version": PACKAGE_VERSION}))


if __name__ == "__main__":
    main()
