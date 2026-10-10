"""Explicit migration of model backends to named contracts and the model runtime.

The router serves every local model through the built-in model runtime
(``provider: model_runtime``). The candle, ONNX Runtime and OpenVINO providers,
their execution fields, the legacy model aliases and the NLI explainer are
gone from the parser; this module rewrites them and records a note for every
change an operator should review.
"""

from __future__ import annotations

import posixpath
from typing import Any
from urllib.parse import urlparse

from cli.config_migration_embeddings import migrate_embedding_models
from cli.config_migration_legacy_models import (
    Replacement,
    is_legacy_mapping,
    is_retired_bundle,
    is_retired_nli,
    replacement_for,
    runtime_artifact,
)
from cli.config_migration_notes import MigrationNotes
from cli.config_migration_paths import (
    as_list,
    binding_maps,
    decisions,
    dict_at,
    signal_maps,
)
from cli.model_runtime_retired import REMOVED_PROVIDERS

MODEL_RUNTIME = "model_runtime"
# Legacy device prefixes and the runtime accelerator that replaces them.
_DEVICE_PREFIXES = {"cuda": "cuda", "rocm": "rocm", "migraphx": "rocm"}
_EXECUTION_FIELDS = ("custom_ops_profile", "compilation_cache_dir")
_LOCAL_SELECTORS = ("use_modernbert", "use_mmbert_32k", "variant")
_HALLUCINATION_NLI_FIELDS = ("enable_nli_filtering", "nli_entailment_threshold")
_HALLUCINATION_ENDPOINT = "hallucination-detector"

# (module path under model_catalog.modules, model field, label-map field).
_MODULE_MODELS = (
    (("classifier", "domain"), "model_id", "category_mapping_path"),
    (("classifier", "pii"), "model_id", "pii_mapping_path"),
    (("prompt_guard",), "model_id", "jailbreak_mapping_path"),
    (("hallucination_mitigation", "fact_check"), "model_id", ""),
    (("feedback_detector",), "model_id", "feedback_mapping_path"),
    (("modality_detector", "classifier"), "model_path", ""),
)


def migrate_prompt_guard_backend(canonical: dict[str, Any]) -> None:
    """Replace the retired prompt_guard.protocol without guessing a service."""

    catalog = canonical.get("global", {}).get("model_catalog", {})
    guard = catalog.get("modules", {}).get("prompt_guard", {})
    if not isinstance(guard, dict) or "protocol" not in guard:
        return
    protocol = guard["protocol"]
    if not protocol:
        guard.pop("protocol")
        return
    field = "global.model_catalog.modules.prompt_guard"
    if protocol not in {"http_chat", "http_classify"}:
        raise ValueError(f"{field}.protocol: unsupported legacy protocol {protocol!r}")
    if guard.get("backend") or guard.get("variant"):
        raise ValueError(f"{field}.protocol conflicts with backend or variant")

    external = catalog.get("external", [])
    candidates = [
        model
        for model in external
        if isinstance(model, dict) and model.get("model_role") == "guardrail"
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{field}.protocol migration requires exactly one guardrail external "
            "model; name the intended service and configure backend explicitly"
        )
    model = candidates[0]
    name = model.get("name") or "guardrail_classifier"
    if any(
        other is not model and other.get("name") == name
        for other in external
        if isinstance(other, dict)
    ):
        raise ValueError(
            f"{field}.protocol migration: external model name {name!r} conflicts; "
            "assign a unique name to the guardrail service"
        )
    model["name"] = name
    guard["backend"] = {
        "protocol": protocol,
        "contract": (
            "label_decision.v1" if protocol == "http_chat" else "label_distribution.v1"
        ),
        "model": name,
    }
    guard.pop("protocol")


def migrate_model_runtime_contract(
    canonical: dict[str, Any], notes: MigrationNotes
) -> None:
    """Move every local model onto the model runtime and retire the NLI paths."""

    catalog = dict_at(canonical, "global", "model_catalog")
    if catalog is not None:
        migrated = _migrate_deployments(catalog, notes)
        _retire_explainer_deployments(catalog, notes)
        for path, bindings in binding_maps(canonical):
            _migrate_bindings(path, bindings, migrated, notes)
        _migrate_system_models(catalog, notes)
        _migrate_modules(catalog, notes)
    migrate_embedding_models(canonical, notes)
    for path, signals in signal_maps(canonical):
        _migrate_signals(path, signals, notes)
    for path, decision in decisions(canonical):
        _migrate_decision(path, decision, notes)
    _migrate_polarity_guard(canonical, notes)


def runtime_device(provider: str, device: Any) -> tuple[str, str | None]:
    """The runtime device for a legacy provider device, and a note if it changed meaning."""

    value = str(device or "").strip() or ("CPU" if provider == "openvino" else "cpu")
    if provider == "openvino":
        if value.upper() == "CPU":
            return "cpu", None
        return "cpu", (
            f"OpenVINO device {value!r} has no runtime equivalent, so the model runs "
            "on CPU; Intel GPUs can use device xpu:N (not yet validated)"
        )
    if value == "cpu":
        return "cpu", None
    prefix, _, index = value.partition(":")
    if prefix == "metal":
        return "mps", None
    if prefix in _DEVICE_PREFIXES and (index or "0").isdigit():
        return f"{_DEVICE_PREFIXES[prefix]}:{index or '0'}", None
    return "cpu", f"unknown legacy device {value!r}; the model runs on CPU"


def _migrate_deployments(catalog: dict[str, Any], notes: MigrationNotes) -> set[str]:
    deployments = catalog.get("deployments")
    if not isinstance(deployments, dict):
        return set()
    migrated: set[str] = set()
    for name, deployment in deployments.items():
        if not isinstance(deployment, dict):
            continue
        path = f"global.model_catalog.deployments.{name}"
        provider = str(deployment.get("provider") or "").strip().lower()
        if provider in REMOVED_PROVIDERS:
            device, problem = runtime_device(provider, deployment.get("device"))
            deployment["provider"] = MODEL_RUNTIME
            deployment["device"] = device
            notes.changed(
                path, f"provider {provider} -> {MODEL_RUNTIME}, device {device}"
            )
            if problem:
                notes.changed(path + ".device", problem)
            _migrate_artifact(path, deployment, notes)
            migrated.add(name)
        elif provider == MODEL_RUNTIME and is_retired_bundle(
            deployment.get("artifact")
        ):
            _migrate_artifact(path, deployment, notes)
        _drop_execution_fields(path, deployment, notes)
    return migrated


def _drop_execution_fields(
    path: str, deployment: dict[str, Any], notes: MigrationNotes
) -> None:
    precision = deployment.pop("precision", None)
    if precision == "fp16":
        deployment.setdefault("profile", "max_speed")
        notes.changed(
            path + ".precision",
            "fp16 -> profile max_speed (the default exact profile runs FP32)",
        )
    elif precision not in (None, "", "native", "fp32"):
        notes.changed(path + ".precision", f"removed unsupported value {precision!r}")
    for field in _EXECUTION_FIELDS:
        if deployment.pop(field, None) not in (None, "", "none"):
            notes.changed(
                f"{path}.{field}", "removed; the runtime selects kernels and graphs"
            )


def _migrate_artifact(
    path: str, deployment: dict[str, Any], notes: MigrationNotes
) -> None:
    artifact = deployment.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        return
    replacement = replacement_for(artifact)
    target = runtime_artifact(artifact)
    if target:
        deployment["artifact"] = target
        if replacement is not None:
            deployment.pop("revision", None)
            notes.changed(
                path + ".artifact", f"{artifact} -> {target}: {replacement.note}"
            )
        elif is_retired_bundle(artifact):
            deployment.pop("revision", None)
            notes.changed(
                path + ".artifact",
                f"{artifact} -> {target} (router images no longer ship prepared "
                "Omni bundles; the runtime serves the published repository)",
            )
        elif target != artifact:
            notes.changed(
                path + ".artifact", f"{artifact} -> {target} (its Hub repository)"
            )
        return
    if posixpath.isabs(artifact) or _looks_like_hub_repository(artifact):
        return
    notes.action(
        path + ".artifact",
        f"{artifact!r} is a relative path; a model_runtime artifact is a Hub "
        "repository or an absolute package path",
    )


def _looks_like_hub_repository(value: str) -> bool:
    owner, slash, name = value.partition("/")
    return (
        bool(slash)
        and bool(owner)
        and bool(name)
        and "/" not in name
        and owner != "models"
    )


def _retire_explainer_deployments(
    catalog: dict[str, Any], notes: MigrationNotes
) -> None:
    deployments = catalog.get("deployments")
    if not isinstance(deployments, dict):
        return
    for name in [key for key, value in deployments.items() if _is_nli(value)]:
        deployments.pop(name)
        notes.changed(
            f"global.model_catalog.deployments.{name}",
            "removed; the NLI explainer is retired",
        )


def _is_nli(deployment: Any) -> bool:
    return isinstance(deployment, dict) and (
        is_retired_nli(deployment.get("artifact"))
        or is_retired_nli(deployment.get("external_model"))
    )


def _migrate_bindings(
    path: str,
    bindings: dict[str, Any],
    migrated_deployments: set[str],
    notes: MigrationNotes,
) -> None:
    if "hallucination_explainer" in bindings:
        bindings.pop("hallucination_explainer")
        notes.changed(
            path + ".hallucination_explainer", "removed; the NLI explainer is retired"
        )
    for consumer, binding in bindings.items():
        if not isinstance(binding, dict):
            continue
        if binding.get("deployment") not in migrated_deployments:
            continue
        head = binding.get("head")
        if isinstance(head, str) and _is_graph_path(head):
            binding.pop("head")
            notes.changed(
                f"{path}.{consumer}.head",
                f"removed graph path {head!r}; the runtime selects the package's graphs",
            )


def _is_graph_path(head: str) -> bool:
    return "/" in head or head.endswith((".onnx", ".xml"))


def _migrate_system_models(catalog: dict[str, Any], notes: MigrationNotes) -> None:
    system = catalog.get("system")
    if not isinstance(system, dict):
        return
    for role, value in list(system.items()):
        path = f"global.model_catalog.system.{role}"
        if role == "hallucination_explainer" or is_retired_nli(value):
            system.pop(role)
            notes.changed(path, "removed; the NLI explainer is retired")
            continue
        replacement = replacement_for(value)
        if replacement is not None:
            system[role] = replacement.target
            notes.changed(path, f"{value} -> {replacement.target}: {replacement.note}")


def _migrate_modules(catalog: dict[str, Any], notes: MigrationNotes) -> None:
    modules = catalog.get("modules")
    if not isinstance(modules, dict):
        return
    base = "global.model_catalog.modules"
    for keys, model_field, mapping_field in _MODULE_MODELS:
        module = dict_at(modules, *keys)
        if module is None:
            continue
        path = ".".join((base, *keys))
        _replace_module_model(path, module, model_field, mapping_field, notes)
        _drop_local_selectors(path, module, notes)
    guard = dict_at(modules, "prompt_guard")
    if guard is not None and guard.pop("model_type", None) not in (None, ""):
        notes.changed(
            base + ".prompt_guard.model_type",
            "removed; the runtime detects the model architecture from the package",
        )
    hallucination = dict_at(modules, "hallucination_mitigation")
    if hallucination is not None:
        _migrate_hallucination(
            base + ".hallucination_mitigation", hallucination, catalog, notes
        )


def _replace_module_model(
    path: str,
    module: dict[str, Any],
    model_field: str,
    mapping_field: str,
    notes: MigrationNotes,
) -> Replacement | None:
    legacy = module.get(model_field)
    replacement = replacement_for(legacy)
    if replacement is None:
        return None
    module[model_field] = replacement.target
    notes.changed(
        f"{path}.{model_field}", f"{legacy} -> {replacement.target}: {replacement.note}"
    )
    if mapping_field and is_legacy_mapping(module.get(mapping_field), legacy):
        module.pop(mapping_field)
        notes.changed(
            f"{path}.{mapping_field}",
            "removed; the router reads the labels of the model it runs",
        )
    return replacement


def _drop_local_selectors(
    path: str, module: dict[str, Any], notes: MigrationNotes
) -> None:
    for field in _LOCAL_SELECTORS:
        if field not in module:
            continue
        value = module.pop(field)
        if value not in (None, False, ""):
            notes.changed(
                f"{path}.{field}",
                "removed; the runtime detects the model architecture from the package",
            )


def _migrate_hallucination(
    path: str,
    hallucination: dict[str, Any],
    catalog: dict[str, Any],
    notes: MigrationNotes,
) -> None:
    for field in ("explainer", "nli_model"):
        if hallucination.pop(field, None) is not None:
            notes.changed(f"{path}.{field}", "removed; the NLI explainer is retired")
    detector = hallucination.get("detector")
    if not isinstance(detector, dict):
        return
    detector_path = path + ".detector"
    for field in _HALLUCINATION_NLI_FIELDS:
        if detector.pop(field, None) not in (None, False):
            notes.changed(
                f"{detector_path}.{field}", "removed; NLI filtering is retired"
            )
    if str(detector.get("backend") or "").strip().lower() == "endpoint":
        _migrate_hallucination_endpoint(detector_path, detector, catalog, notes)
        return
    if detector.pop("endpoint", None) is not None:
        notes.changed(
            detector_path + ".endpoint",
            "removed; a remote detector is a hallucination_detector binding",
        )
    if detector.pop("backend", None) not in (None, ""):
        notes.changed(
            detector_path + ".backend",
            "removed; the detector runs in the model runtime",
        )
    _replace_module_model(detector_path, detector, "model_id", "", notes)
    if detector.pop("include_explanation", None):
        notes.changed(
            detector_path + ".include_explanation",
            "removed; NLI explanations are retired, spans stay in the response",
        )


def _migrate_hallucination_endpoint(
    path: str, detector: dict[str, Any], catalog: dict[str, Any], notes: MigrationNotes
) -> None:
    """Rewrite `backend: endpoint` into the http_chat binding it was shorthand for.

    The shorthand applied to every recipe without its own binding, so the
    binding becomes the catalog-wide default.
    """
    endpoint = str(detector.get("endpoint") or "").strip()
    model = str(detector.get("model_id") or "").strip()
    parsed = urlparse(endpoint.rstrip("/"))
    if not model or parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(
            f"{path}: backend: endpoint needs an absolute http(s) endpoint and a model_id"
        )
    if parsed.path != "/v1" or parsed.query or parsed.fragment:
        raise ValueError(
            f"{path}.endpoint {endpoint!r}: an http_chat deployment posts to "
            "<host>/v1/chat/completions; declare the external model and the "
            "hallucination_detector binding by hand for any other path"
        )
    bindings = catalog.setdefault("bindings", {})
    if "hallucination_detector" in bindings:
        notes.changed(
            path + ".backend",
            "removed; the hallucination_detector binding already selects the detector",
        )
    else:
        deployments = catalog.setdefault("deployments", {})
        external = catalog.setdefault("external", [])
        name = _HALLUCINATION_ENDPOINT
        if name in deployments or any(
            isinstance(other, dict) and other.get("name") == name for other in external
        ):
            raise ValueError(
                f"{path}: the name {name!r} is taken; declare the "
                "hallucination_detector binding by hand"
            )
        external.append(
            {
                "name": name,
                "model_role": "classification",
                "llm_endpoint": {
                    "address": parsed.hostname,
                    "port": parsed.port or (443 if parsed.scheme == "https" else 80),
                    "protocol": parsed.scheme,
                },
                "llm_model_name": model,
                "llm_timeout_seconds": 10,
            }
        )
        deployments[name] = {"provider": "http", "external_model": name}
        bindings["hallucination_detector"] = {
            "deployment": name,
            "contract": "token_spans.v1",
            "adapter": "http_chat",
        }
        notes.changed(
            path + ".backend",
            "moved to global.model_catalog.bindings.hallucination_detector "
            f"(http_chat deployment {name})",
        )
    for field in ("backend", "endpoint", "model_id"):
        detector.pop(field, None)


def _migrate_signals(path: str, signals: dict[str, Any], notes: MigrationNotes) -> None:
    for index, rule in enumerate(as_list(signals.get("classifiers"))):
        if not isinstance(rule, dict):
            continue
        legacy = rule.get("model_path")
        replacement = replacement_for(legacy)
        if replacement is not None:
            rule["model_path"] = replacement.target
            notes.changed(
                f"{path}.classifiers[{index}].model_path",
                f"{legacy} -> {replacement.target}: {replacement.note}",
            )
    for index, rule in enumerate(as_list(signals.get("hallucination"))):
        if isinstance(rule, dict) and rule.pop("use_nli", None) is not None:
            notes.changed(
                f"{path}.hallucination[{index}].use_nli",
                "removed; NLI explanations are retired",
            )


def _migrate_decision(
    path: str, decision: dict[str, Any], notes: MigrationNotes
) -> None:
    for index, plugin in enumerate(as_list(decision.get("plugins"))):
        if not isinstance(plugin, dict) or plugin.get("type") != "hallucination":
            continue
        configuration = plugin.get("configuration")
        if (
            isinstance(configuration, dict)
            and configuration.pop("use_nli", None) is not None
        ):
            notes.changed(
                f"{path}.plugins[{index}].configuration.use_nli",
                "removed; NLI explanations are retired",
            )
    mlp = dict_at(decision, "algorithm", "mlp")
    if mlp is not None and mlp.pop("device", None) is not None:
        notes.changed(
            path + ".algorithm.mlp.device", "removed; the MLP selector runs in Go"
        )
    grounding = dict_at(decision, "algorithm", "fusion", "grounding")
    if grounding is not None and "nli_contradiction_penalty" in grounding:
        penalty = grounding.pop("nli_contradiction_penalty")
        grounding.setdefault("contradiction_penalty", penalty)
        notes.changed(
            path + ".algorithm.fusion.grounding.nli_contradiction_penalty",
            "-> contradiction_penalty; grounding reads the hallucination detector, "
            "not an NLI model",
        )


def _migrate_polarity_guard(canonical: dict[str, Any], notes: MigrationNotes) -> None:
    cache = dict_at(canonical, "global", "stores", "response_cache")
    if cache is not None and "polarity_guard" in cache:
        cache.pop("polarity_guard")
        notes.changed(
            "global.stores.response_cache.polarity_guard",
            "removed; the NLI tier is retired and the lexical guard always runs",
        )
