#!/usr/bin/env python3
"""Verify the self-contained MoM v0alpha1 model-format fixture."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:
    import yaml
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing verifier dependency. Install jsonschema and PyYAML, then rerun."
    ) from exc


ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "example"
TERMINAL_STATUSES = {"ok", "abstain", "reject", "error"}
SCHEMA_FILES = {
    "manifest": "manifest.schema.json",
    "ir": "ir.schema.json",
    "realization": "realization-profile.schema.json",
    "binding": "binding.schema.json",
    "lock": "lock.schema.json",
    "eval_protocol": "eval-protocol.schema.json",
    "eval_attestation": "eval-attestation.schema.json",
    "run_record": "run-record.schema.json",
}


class VerificationError(RuntimeError):
    """A model-format invariant did not hold."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON at {path}: {exc}") from exc


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise VerificationError(f"invalid YAML at {path}: {exc}") from exc


def load_data(path: Path) -> Any:
    return load_json(path) if path.suffix == ".json" else load_yaml(path)


def load_schemas() -> dict[str, dict[str, Any]]:
    schemas = {name: load_json(ROOT / file) for name, file in SCHEMA_FILES.items()}
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)
    return schemas


def validate_schema(
    name: str, value: Any, schemas: dict[str, dict[str, Any]]
) -> None:
    errors = sorted(
        Draft202012Validator(
            schemas[name], format_checker=FormatChecker()
        ).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise VerificationError(f"{name} schema validation failed: {details}")


def resolve_regular_file(root: Path, relative: str) -> Path:
    """Resolve a normalized POSIX path without traversal or symlinks."""
    if not isinstance(relative, str) or not relative:
        raise VerificationError("object path must be a nonempty string")
    if "\\" in relative:
        raise VerificationError(f"backslash is forbidden in object path: {relative}")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise VerificationError(f"unsafe object path: {relative}")
    if str(posix) != relative:
        raise VerificationError(f"object path is not normalized: {relative}")

    candidate = root.joinpath(*posix.parts)
    current = root
    for part in posix.parts:
        current = current / part
        if current.is_symlink():
            raise VerificationError(f"symlink is forbidden in object path: {relative}")
    if not candidate.is_file():
        raise VerificationError(f"referenced object is not a regular file: {relative}")
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if (
        resolved_candidate.parent != resolved_root
        and resolved_root not in resolved_candidate.parents
    ):
        raise VerificationError(f"object escaped root: {relative}")
    return candidate


def descriptor_without_role(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        key: descriptor[key] for key in ("path", "mediaType", "digest", "size")
    }


def verify_descriptor(root: Path, descriptor: dict[str, Any]) -> Path:
    path = resolve_regular_file(root, descriptor["path"])
    actual_digest = sha256_file(path)
    actual_size = path.stat().st_size
    if actual_digest != descriptor["digest"]:
        raise VerificationError(
            f"digest mismatch for {descriptor['path']}: "
            f"expected {descriptor['digest']}, got {actual_digest}"
        )
    if actual_size != descriptor["size"]:
        raise VerificationError(
            f"size mismatch for {descriptor['path']}: "
            f"expected {descriptor['size']}, got {actual_size}"
        )
    return path


def check_negative_paths(schemas: dict[str, dict[str, Any]]) -> None:
    validator = Draft202012Validator(schemas["manifest"]["$defs"]["relativePath"])
    unsafe = ["/absolute", "../escape", "a/../escape", "a/./b", "a//b", "a\\b"]
    accepted = [path for path in unsafe if validator.is_valid(path)]
    if accepted:
        raise VerificationError(f"path schema accepted unsafe cases: {accepted}")


def manifest_descriptors(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [manifest["specification"], *manifest["objects"]]


def check_manifest(
    manifest: dict[str, Any], schemas: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], str, str]:
    validate_schema("manifest", manifest, schemas)
    no_secret_values(manifest)
    descriptors = manifest_descriptors(manifest)
    paths = [descriptor["path"] for descriptor in descriptors]
    if len(paths) != len(set(paths)):
        raise VerificationError("manifest contains duplicate object paths")

    loaded: dict[str, Any] = {}
    for descriptor in descriptors:
        path = verify_descriptor(EXAMPLE, descriptor)
        if path.suffix in {".json", ".yaml", ".yml"}:
            loaded[descriptor["path"]] = load_data(path)
    for value in loaded.values():
        no_secret_values(value)

    ir_path = verify_descriptor(EXAMPLE, manifest["specification"])
    ir = load_json(ir_path)
    validate_schema("ir", ir, schemas)
    no_secret_values(ir)
    semantic_digest = sha256_bytes(canonical_json_bytes(ir))
    if semantic_digest != manifest["semanticDigest"]:
        raise VerificationError("semanticDigest is not the canonical typed-IR digest")
    bundle_digest = sha256_bytes(canonical_json_bytes(manifest))

    if manifest["model"] != ir["model"]:
        raise VerificationError("manifest and IR model coordinates differ")
    if manifest["entrypoints"] != ir["entrypoints"]:
        raise VerificationError("manifest and IR entrypoints differ")

    behavior_id = manifest["model"]["behaviorVariant"]
    if {item["id"] for item in ir["behaviorVariants"]} != {behavior_id}:
        raise VerificationError("fixture must expose exactly its declared behavior variant")
    if any(item["behaviorVariant"] != behavior_id for item in ir["entrypoints"]):
        raise VerificationError("entrypoint behavior variant differs from model identity")

    ir_assets = {item["path"]: item for item in ir["artifacts"].values()}
    manifest_assets = {
        item["path"]: descriptor_without_role(item)
        for item in manifest["objects"]
        if item["role"] in {"semantic-asset", "contract"}
    }
    if ir_assets != manifest_assets:
        raise VerificationError(
            "IR semantic-asset closure differs from manifest semantic objects"
        )

    for item in manifest["objects"]:
        if item["role"] == "eval-protocol":
            protocol = loaded[item["path"]]
            no_secret_values(protocol)
            validate_schema("eval_protocol", protocol, schemas)
            if protocol["behaviorVariant"] != behavior_id:
                raise VerificationError("evaluation protocol targets another behavior")
        elif item["role"] == "realization-profile":
            no_secret_values(loaded[item["path"]])
            validate_schema("realization", loaded[item["path"]], schemas)

    roles = {item["role"] for item in manifest["objects"]}
    if "eval-protocol" not in roles or "realization-profile" not in roles:
        raise VerificationError(
            "fixture must include an evaluation protocol and realization profiles"
        )
    return ir, semantic_digest, bundle_digest


def operator_successors(operator: dict[str, Any]) -> list[str]:
    if operator["op"] == "select":
        return [operator["next"]]
    if operator["op"] == "contract-check":
        return [operator["onSuccess"], operator["onFailure"]]
    return []


def check_graph(ir: dict[str, Any]) -> None:
    named_collections = (
        (ir["entrypoints"], "id", "entrypoint"),
        (ir["logicalModels"], "id", "logical model"),
        (ir["signals"], "name", "signal"),
        (ir["projections"], "name", "projection"),
        (ir["behaviorVariants"], "id", "behavior variant"),
    )
    for items, key, label in named_collections:
        values = [item[key] for item in items]
        if len(values) != len(set(values)):
            raise VerificationError(f"{label} identifiers are not unique")

    known_features = {signal["name"] for signal in ir["signals"]}
    for projection in ir["projections"]:
        if not set(projection["inputs"]).issubset(known_features):
            raise VerificationError(
                f"projection {projection['name']} names an unknown or forward input"
            )
        known_features.add(projection["name"])

    operators = {operator["id"]: operator for operator in ir["operators"]}
    if len(operators) != len(ir["operators"]):
        raise VerificationError("operator identifiers are not unique")
    entry = ir["graph"]["entry"]
    if entry not in operators:
        raise VerificationError("graph entry does not name an operator")
    for entrypoint in ir["entrypoints"]:
        if entrypoint["graphEntry"] not in operators:
            raise VerificationError(
                f"entrypoint {entrypoint['id']} names a missing graph entry"
            )

    logical_models = {model["id"] for model in ir["logicalModels"]}
    artifacts = set(ir["artifacts"])
    for operator in operators.values():
        for successor in operator_successors(operator):
            if successor not in operators:
                raise VerificationError(
                    f"operator {operator['id']} names missing successor {successor}"
                )
        if operator["op"] == "select":
            if not set(operator["candidates"]).issubset(logical_models):
                raise VerificationError("select operator names an unknown logical model")
            for field in ("policyArtifact", "promptArtifact"):
                if field in operator and operator[field] not in artifacts:
                    raise VerificationError(
                        f"select operator names unknown artifact {operator[field]}"
                    )
        if (
            operator["op"] == "contract-check"
            and operator["contractArtifact"] not in artifacts
        ):
            raise VerificationError("contract operator names an unknown contract")
        if operator["op"] == "return":
            if operator["status"] not in TERMINAL_STATUSES:
                raise VerificationError("unknown terminal status")
            if operator["status"] != "ok" and "valueFrom" in operator:
                raise VerificationError("non-answer return carries an answer value")

    reached: set[str] = set()
    stack = [entry]
    while stack:
        current = stack.pop()
        if current in reached:
            continue
        reached.add(current)
        stack.extend(operator_successors(operators[current]))
    if reached != set(operators):
        raise VerificationError("operator graph contains unreachable nodes")
    if not any(operators[node]["op"] == "return" for node in reached):
        raise VerificationError("operator graph has no reachable terminal")

    state: dict[str, int] = {}

    def prove_termination(node: str) -> None:
        if state.get(node) == 1:
            raise VerificationError("v0alpha1 core graph contains a cycle")
        if state.get(node) == 2:
            return
        operator = operators[node]
        if operator["op"] == "return":
            state[node] = 2
            return
        successors = operator_successors(operator)
        if not successors:
            raise VerificationError(
                f"nonterminal operator {node} has no declared successor"
            )
        state[node] = 1
        for successor in successors:
            prove_termination(successor)
        state[node] = 2

    prove_termination(entry)

    bounds = ir["bounds"]
    if len(reached) > bounds["maxEvents"]:
        raise VerificationError("fixture graph exceeds its event bound")

    constraints = [
        constraint
        for profile in ir["behaviorVariants"]
        for constraint in profile["hardConstraints"]
    ]
    if not constraints or any(
        constraint.get("onUnknown") != "reject" for constraint in constraints
    ):
        raise VerificationError("hard constraints are absent or not fail closed")
    for variant in ir["behaviorVariants"]:
        if not preference_is_valid(
            variant["requestPreferenceDomain"]["default"], variant
        ):
            raise VerificationError("behavior variant has an invalid preference domain")
        for constraint in variant["hardConstraints"]:
            signal = constraint.get("when", {}).get("signal")
            if signal is not None and signal not in known_features:
                raise VerificationError("hard constraint names an unknown signal")


def no_secret_values(value: Any, path: str = "") -> None:
    forbidden = {
        "apikey",
        "api_key",
        "password",
        "secretvalue",
        "secret_value",
        "credential",
        "credentials",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in forbidden:
                raise VerificationError(f"secret-bearing field found at {path}/{key}")
            no_secret_values(nested, f"{path}/{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            no_secret_values(nested, f"{path}/{index}")


def subset(required: Iterable[str], actual: Iterable[str]) -> bool:
    return set(required).issubset(set(actual))


def version_satisfies(actual: str, requirement: str) -> bool:
    """Evaluate the small numeric-version range used by the fixture."""
    if not any(symbol in requirement for symbol in (">", "<", "=")):
        return actual == requirement
    try:
        def numeric_version(value: str) -> tuple[int, ...]:
            parts = [int(part) for part in value.split(".")]
            return tuple(parts + [0] * max(0, 3 - len(parts)))

        actual_version = numeric_version(actual)
        for clause in requirement.split(","):
            clause = clause.strip()
            operator = next(
                token for token in (">=", "<=", "==", ">", "<")
                if clause.startswith(token)
            )
            required_version = numeric_version(clause[len(operator):])
            comparisons = {
                ">=": actual_version >= required_version,
                "<=": actual_version <= required_version,
                "==": actual_version == required_version,
                ">": actual_version > required_version,
                "<": actual_version < required_version,
            }
            if not comparisons[operator]:
                return False
        return True
    except (ValueError, StopIteration):
        return False


def preference_is_valid(
    preference: dict[str, float], variant: dict[str, Any]
) -> bool:
    domain = variant["requestPreferenceDomain"]
    parameters = domain["parameters"]
    if set(domain["default"]) != set(parameters) or set(preference) != set(parameters):
        return False
    return all(
        specification["minimum"]
        <= preference[name]
        <= specification["maximum"]
        and specification["minimum"] <= specification["maximum"]
        for name, specification in parameters.items()
    )


def candidate_satisfies(
    candidate: dict[str, Any], logical_model: dict[str, Any]
) -> bool:
    requirements = logical_model["requirements"]
    capabilities = candidate["capabilities"]
    identity = logical_model["identity"]
    artifact = candidate["artifact"]
    return (
        artifact["uri"] == identity["uri"]
        and artifact.get("revision") == identity.get("revision")
        and artifact["identityStrength"] == identity["strength"]
        and subset(requirements["protocols"], capabilities["protocols"])
        and subset(requirements["modalities"], capabilities["modalities"])
        and subset(requirements["requiredFeatures"], capabilities["features"])
        and capabilities["maxInputContextTokens"]
        >= requirements["minInputContextTokens"]
        and capabilities["maxOutputTokens"] >= requirements["minOutputTokens"]
        and subset(requirements["dataPolicies"], capabilities["dataPolicies"])
    )


def candidate_matches_profile(
    candidate: dict[str, Any], target: dict[str, Any]
) -> bool:
    if candidate["modelRuntime"]["name"] not in target["modelRuntimes"]:
        return False
    if candidate["deployment"]["driver"] not in target["deploymentDrivers"]:
        return False
    if candidate["artifact"]["format"] not in target["artifactFormats"]:
        return False
    if candidate["wireTransport"]["protocol"] not in target["wireTransports"]:
        return False
    if candidate["apiAdapter"]["interface"] not in target["apiAdapters"]:
        return False
    required_accelerator = target.get("accelerator")
    actual_accelerator = candidate.get("accelerator")
    if required_accelerator:
        return bool(
            actual_accelerator
            and actual_accelerator["api"] in required_accelerator["apis"]
            and actual_accelerator["vendor"] in required_accelerator["vendors"]
            and (
                not required_accelerator.get("architectures")
                or actual_accelerator["architecture"]
                in required_accelerator["architectures"]
            )
        )
    return True


def find_realization_profile(
    manifest: dict[str, Any], profile_id: str, schemas: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    for descriptor in manifest["objects"]:
        if descriptor["role"] != "realization-profile":
            continue
        profile = load_yaml(resolve_regular_file(EXAMPLE, descriptor["path"]))
        validate_schema("realization", profile, schemas)
        if profile["id"] == profile_id:
            return profile
    raise VerificationError(f"binding selects unknown realization profile {profile_id}")


def check_binding(
    binding: dict[str, Any],
    manifest: dict[str, Any],
    ir: dict[str, Any],
    semantic_digest: str,
    bundle_digest: str,
    schemas: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    validate_schema("binding", binding, schemas)
    no_secret_values(binding)
    expected_subject = {
        "modelId": manifest["model"]["id"],
        "modelVersion": manifest["model"]["version"],
        "behaviorVariant": manifest["model"]["behaviorVariant"],
        "semanticDigest": semantic_digest,
        "bundleDigest": bundle_digest,
    }
    if binding["subject"] != expected_subject:
        raise VerificationError("binding subject differs from bundle identity")

    profile = find_realization_profile(
        manifest, binding["realizationProfile"], schemas
    )
    if binding["controlEngine"]["name"] not in profile["controlEngine"]["names"]:
        raise VerificationError("binding control engine violates realization profile")
    if not subset(
        profile["controlEngine"]["requiredCapabilities"],
        manifest["compatibility"]["requiredCapabilities"],
    ):
        raise VerificationError(
            "realization profile requires capabilities absent from bundle compatibility"
        )
    logical_models = {model["id"]: model for model in ir["logicalModels"]}
    if set(binding["modelBindings"]) != set(logical_models):
        raise VerificationError("binding does not cover the logical model set exactly")
    if set(profile["targets"]) != set(logical_models):
        raise VerificationError("realization profile target set differs from IR")

    for model_id, logical_model in logical_models.items():
        candidates = binding["modelBindings"][model_id]["candidates"]
        if len({candidate["id"] for candidate in candidates}) != len(candidates):
            raise VerificationError(f"candidate identifiers repeat for {model_id}")
        for candidate in candidates:
            for header in candidate["wireTransport"].get("headers", []):
                if header["name"].lower() in {
                    "authorization",
                    "api-key",
                    "x-api-key",
                } and "value" in header:
                    raise VerificationError(
                        f"candidate {candidate['id']} embeds a secret header value"
                    )
            if not candidate_satisfies(candidate, logical_model):
                raise VerificationError(
                    f"candidate {candidate['id']} violates {model_id} semantics"
                )
            if not candidate_matches_profile(candidate, profile["targets"][model_id]):
                raise VerificationError(
                    f"candidate {candidate['id']} violates realization profile"
                )
            for evidence_block in ("capacity", "pricing"):
                block = candidate[evidence_block]
                if block["status"] == "known" and len(block) == 1:
                    raise VerificationError(
                        f"candidate {candidate['id']} marks empty "
                        f"{evidence_block} evidence as known"
                    )

    binding_digest = sha256_bytes(canonical_json_bytes(binding))
    return binding_digest, profile


def check_lock(
    lock: dict[str, Any],
    binding: dict[str, Any],
    binding_digest: str,
    manifest: dict[str, Any],
    profile: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
) -> str:
    validate_schema("lock", lock, schemas)
    no_secret_values(lock)
    if lock["subject"] != binding["subject"]:
        raise VerificationError("lock subject differs from binding")
    if lock["bindingDigest"] != binding_digest:
        raise VerificationError("lock bindingDigest differs from canonical binding")
    if lock["realizationProfile"] != binding["realizationProfile"]:
        raise VerificationError("lock realization profile differs from binding")
    if "floating" in json.dumps(lock):
        raise VerificationError("production fixture contains a floating revision")
    if set(lock["targets"]) != set(binding["modelBindings"]):
        raise VerificationError("lock target set differs from binding target set")
    if lock["controlEngine"]["name"] != binding["controlEngine"]["name"]:
        raise VerificationError("lock control engine differs from binding")
    if lock["controlEngine"]["resolver"] != binding["controlEngine"]["resolver"]:
        raise VerificationError("lock resolver differs from binding")
    required_capabilities = set(manifest["compatibility"]["requiredCapabilities"])
    required_capabilities.update(
        profile["controlEngine"]["requiredCapabilities"]
    )
    if not required_capabilities.issubset(
        set(lock["controlEngine"]["supportedCapabilities"])
    ):
        raise VerificationError("locked engine lacks required capabilities")
    supported_opsets = lock["controlEngine"]["supportedOpsets"]
    for required_opset in manifest["compatibility"]["opsets"]:
        resolved_versions = [
            item.split("@", 1)[1]
            for item in supported_opsets
            if item.startswith(required_opset["name"] + "@")
        ]
        if not any(
            version_satisfies(version, required_opset["version"])
            for version in resolved_versions
        ):
            raise VerificationError("locked engine lacks a required opset")

    for model_id, locked_target in lock["targets"].items():
        binding_candidates = {
            item["id"]: item
            for item in binding["modelBindings"][model_id]["candidates"]
        }
        locked_ids = {item["bindingId"] for item in locked_target["candidates"]}
        if locked_ids != set(binding_candidates):
            raise VerificationError(
                f"lock candidate inventory differs from binding for {model_id}"
            )
        for candidate in locked_target["candidates"]:
            binding_candidate = binding_candidates[candidate["bindingId"]]
            artifact = candidate["artifact"]
            if artifact["uri"] != binding_candidate["artifact"]["uri"]:
                raise VerificationError("locked artifact URI differs from binding")
            if artifact["format"] != binding_candidate["artifact"]["format"]:
                raise VerificationError("locked artifact format differs from binding")
            if artifact["identityStrength"] != binding_candidate["artifact"][
                "identityStrength"
            ]:
                raise VerificationError(
                    "locked artifact identity strength differs from binding"
                )
            if artifact.get("revision") != binding_candidate["artifact"].get(
                "revision"
            ):
                raise VerificationError("locked artifact revision differs from binding")
            if candidate["wireTransport"]["protocol"] != binding_candidate[
                "wireTransport"
            ]["protocol"]:
                raise VerificationError("locked wire transport differs from binding")
            if candidate["apiAdapter"] != binding_candidate["apiAdapter"]:
                raise VerificationError("locked API adapter differs from binding")
            if candidate.get("deploymentDriver") != binding_candidate["deployment"][
                "driver"
            ]:
                raise VerificationError("locked deployment driver differs from binding")
            if candidate["runtime"]["name"] != binding_candidate["modelRuntime"][
                "name"
            ]:
                raise VerificationError("locked runtime differs from binding")
            if not version_satisfies(
                candidate["runtime"]["version"],
                binding_candidate["modelRuntime"]["version"],
            ):
                raise VerificationError("locked runtime version violates binding")
            bound_image = binding_candidate["modelRuntime"].get("image")
            if bound_image:
                expected_image_digest = (
                    bound_image.rsplit("@", 1)[1] if "@" in bound_image else None
                )
                if candidate["runtime"].get("imageDigest") != expected_image_digest:
                    raise VerificationError("locked runtime image differs from binding")
            bound_accelerator = binding_candidate.get("accelerator")
            locked_accelerator = candidate.get("accelerator")
            if bool(bound_accelerator) != bool(locked_accelerator):
                raise VerificationError("locked accelerator presence differs from binding")
            if bound_accelerator and any(
                locked_accelerator[key] != bound_accelerator[key]
                for key in ("api", "vendor", "architecture")
            ):
                raise VerificationError("locked accelerator differs from binding")
            strength = artifact["identityStrength"]
            if strength in {
                "content-addressed",
                "repository-revision",
                "provider-version",
            } and not artifact.get("revision"):
                raise VerificationError("pinned lock artifact lacks revision")
            if strength == "observed-opaque":
                if not (
                    binding["policy"]["allowOpaqueProviderRevisions"]
                    and artifact.get("observedAlias")
                    and artifact.get("observedAt")
                ):
                    raise VerificationError("opaque lock evidence is incomplete")
            verify_descriptor(EXAMPLE, candidate["capabilityEvidence"])

    return sha256_bytes(canonical_json_bytes(lock))


def assert_subject(
    subject: dict[str, Any],
    semantic_digest: str,
    bundle_digest: str,
    binding_digest: str,
    lock_digest: str,
) -> None:
    expected = {
        "semanticDigest": semantic_digest,
        "bundleDigest": bundle_digest,
        "bindingDigest": binding_digest,
        "lockDigest": lock_digest,
    }
    for key, value in expected.items():
        if subject.get(key) != value:
            raise VerificationError(f"evidence subject has incorrect {key}")


def check_evidence(
    semantic_digest: str,
    bundle_digest: str,
    binding_digest: str,
    lock_digest: str,
    manifest: dict[str, Any],
    ir: dict[str, Any],
    lock: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
) -> None:
    attestation = load_json(EXAMPLE / "evidence/eval-attestation.json")
    validate_schema("eval_attestation", attestation, schemas)
    no_secret_values(attestation)
    assert_subject(
        attestation["subject"],
        semantic_digest,
        bundle_digest,
        binding_digest,
        lock_digest,
    )
    if "quality, cost, latency, or portability result" not in attestation[
        "claimBoundary"
    ]:
        raise VerificationError("fixture attestation lacks a narrow claim boundary")

    protocol_descriptor = next(
        item for item in manifest["objects"] if item["role"] == "eval-protocol"
    )
    protocol = load_yaml(EXAMPLE / protocol_descriptor["path"])
    if attestation["suite"]["protocolDigest"] != protocol_descriptor["digest"]:
        raise VerificationError("attestation protocol digest differs from bundle")
    protocol_datasets = {
        task["id"]: task["dataset"]["digest"] for task in protocol["tasks"]
    }
    attested_datasets = {
        item["taskId"]: item["datasetDigest"]
        for item in attestation["suite"]["datasets"]
    }
    if attested_datasets != protocol_datasets:
        raise VerificationError("attestation dataset set differs from protocol")
    for key in ("modelId", "modelVersion", "behaviorVariant"):
        expected = (
            manifest["model"]["behaviorVariant"]
            if key == "behaviorVariant"
            else manifest["model"]["id" if key == "modelId" else "version"]
        )
        if attestation["subject"][key] != expected:
            raise VerificationError(f"attestation subject has incorrect {key}")
    for key in ("seed", "concurrency", "temperature"):
        if attestation["execution"][key] != protocol["execution"][key]:
            raise VerificationError(f"attestation execution has incorrect {key}")
    protocol_metrics = {
        task["id"]: set(task["metrics"]) for task in protocol["tasks"]
    }
    for metric in attestation["metrics"]:
        if metric["taskId"] not in protocol_metrics or metric["name"] not in (
            protocol_metrics[metric["taskId"]]
        ):
            raise VerificationError(
                "attestation reports a task metric absent from protocol"
            )
        if protocol["reporting"]["includeConfidenceIntervals"] and (
            "confidenceInterval" not in metric
        ):
            raise VerificationError("attestation omits a required confidence interval")

    run_record = load_json(EXAMPLE / "evidence/run-record.json")
    validate_schema("run_record", run_record, schemas)
    no_secret_values(run_record)
    assert_subject(
        run_record["subject"],
        semantic_digest,
        bundle_digest,
        binding_digest,
        lock_digest,
    )
    sequence = [event["sequence"] for event in run_record["events"]]
    if sequence != list(range(len(sequence))):
        raise VerificationError("run-record event sequence is not canonical")
    if run_record["entrypoint"] not in {
        entrypoint["id"] for entrypoint in ir["entrypoints"]
    }:
        raise VerificationError("run record names an unknown entrypoint")
    variants = {variant["id"]: variant for variant in ir["behaviorVariants"]}
    if run_record["behaviorVariant"] != manifest["model"]["behaviorVariant"]:
        raise VerificationError("run record behavior variant differs from model")
    if not preference_is_valid(
        run_record["requestPreference"], variants[run_record["behaviorVariant"]]
    ):
        raise VerificationError("run record request preference is outside its domain")
    operators = {operator["id"]: operator for operator in ir["operators"]}
    events_by_id: dict[str, dict[str, Any]] = {}
    seen_events: set[str] = set()
    for event in run_record["events"]:
        if event["id"] in seen_events:
            raise VerificationError("run-record event identifiers repeat")
        if not set(event["parentIds"]).issubset(seen_events):
            raise VerificationError("run-record event names a non-prior parent")
        if "retryOf" in event and event["retryOf"] not in seen_events:
            raise VerificationError("run-record retry does not name a prior event")
        if event["operator"] not in operators:
            raise VerificationError("run-record event names an unknown operator")
        for parent_id in event["parentIds"]:
            parent_operator = operators[events_by_id[parent_id]["operator"]]
            if event["operator"] not in operator_successors(parent_operator):
                raise VerificationError("run-record edge violates the typed IR graph")
        if "logicalModel" in event:
            logical_model = event["logicalModel"]
            if logical_model not in lock["targets"]:
                raise VerificationError("run event names an unlocked logical model")
            locked_candidates = {
                candidate["bindingId"]: candidate
                for candidate in lock["targets"][logical_model]["candidates"]
            }
            if event.get("bindingId") not in locked_candidates:
                raise VerificationError("run event names an unlocked binding candidate")
            expected_observed_id = locked_candidates[event["bindingId"]][
                "apiAdapter"
            ]["externalModelId"]
            if event.get("observedModelId") != expected_observed_id:
                raise VerificationError("run event observed model differs from lock")
        seen_events.add(event["id"])
        events_by_id[event["id"]] = event

    final_event = run_record["events"][-1]
    final_operator = operators[final_event["operator"]]
    if final_operator["op"] != "return" or final_operator["status"] != run_record[
        "terminalStatus"
    ]:
        raise VerificationError("run terminal status differs from typed IR")

    provenance = load_json(EXAMPLE / "evidence/build-provenance.json")
    no_secret_values(provenance)
    if provenance["subject"] != {
        "modelId": manifest["model"]["id"],
        "semanticDigest": semantic_digest,
        "bundleDigest": bundle_digest,
    }:
        raise VerificationError("build provenance subject differs from bundle")


def main() -> int:
    schemas = load_schemas()
    check_negative_paths(schemas)
    manifest = load_json(EXAMPLE / "manifest.json")
    ir, semantic_digest, bundle_digest = check_manifest(manifest, schemas)
    check_graph(ir)

    binding = load_yaml(EXAMPLE / "deployment.binding.yaml")
    binding_digest, profile = check_binding(
        binding, manifest, ir, semantic_digest, bundle_digest, schemas
    )
    lock = load_json(EXAMPLE / "resolution.lock.json")
    lock_digest = check_lock(
        lock, binding, binding_digest, manifest, profile, schemas
    )
    check_evidence(
        semantic_digest,
        bundle_digest,
        binding_digest,
        lock_digest,
        manifest,
        ir,
        lock,
        schemas,
    )

    print("MoM model-format fixture verification passed")
    print(f"semanticDigest {semantic_digest}")
    print(f"bundleDigest   {bundle_digest}")
    print(f"bindingDigest  {binding_digest}")
    print(f"lockDigest     {lock_digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
