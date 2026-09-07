"""Cross-language semantic identity for the immutable run manifest."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any

_GO_JSON_SCIENTIFIC_LOWER_BOUND = 1e-6
_GO_JSON_SCIENTIFIC_UPPER_BOUND = 1e21


def _field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _optional_field(value: object, name: str) -> Any | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name)


def _go_timestamp(value: datetime | str) -> str:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("manifest created_at must include a timezone")
    encoded = parsed.isoformat(timespec="microseconds")
    head, offset = encoded[:-6], encoded[-6:]
    if "." in head:
        head = head.rstrip("0").rstrip(".")
    encoded = head + offset
    return encoded[:-6] + "Z" if encoded.endswith("+00:00") else encoded


def _model_arm_value(arm: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": _field(arm, "id"),
        "model": _field(arm, "model"),
        "provider_model_id_digest": _field(arm, "provider_model_id_digest"),
        "input_cost_per_million_tokens_usd": _field(
            arm, "input_cost_per_million_tokens_usd"
        ),
        "output_cost_per_million_tokens_usd": _field(
            arm, "output_cost_per_million_tokens_usd"
        ),
        "capabilities": list(_optional_field(arm, "capabilities") or ()),
        "modalities": list(_optional_field(arm, "modalities") or ()),
    }
    for name in (
        "context_window_tokens",
        "parameter_size",
        "runtime_revision",
        "config_digest",
    ):
        optional = _optional_field(arm, name)
        if optional is not None:
            value[name] = optional
    return value


def _model_arm_snapshot_value(arm: object) -> OrderedDict[str, object]:
    """Match the declared Go ``ModelArm`` order and ``omitempty`` behavior."""

    value: OrderedDict[str, object] = OrderedDict(
        (
            ("id", _field(arm, "id")),
            ("model", _field(arm, "model")),
            (
                "provider_model_id_digest",
                _field(arm, "provider_model_id_digest"),
            ),
            (
                "input_cost_per_million_tokens_usd",
                _field(arm, "input_cost_per_million_tokens_usd"),
            ),
            (
                "output_cost_per_million_tokens_usd",
                _field(arm, "output_cost_per_million_tokens_usd"),
            ),
        )
    )
    capabilities = tuple(_optional_field(arm, "capabilities") or ())
    modalities = tuple(_optional_field(arm, "modalities") or ())
    if capabilities:
        value["capabilities"] = list(capabilities)
    if modalities:
        value["modalities"] = list(modalities)
    for name in (
        "context_window_tokens",
        "parameter_size",
        "runtime_revision",
        "config_digest",
    ):
        optional = _optional_field(arm, name)
        if optional is not None:
            value[name] = optional
    return value


def _support_model_value(model: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model": _field(model, "model"),
        "provider_model_id_digest": _field(model, "provider_model_id_digest"),
        "config_digest": _field(model, "config_digest"),
        "backend_topology_digest": _field(model, "backend_topology_digest"),
    }
    runtime_revision = _optional_field(model, "runtime_revision")
    if runtime_revision is not None:
        value["runtime_revision"] = runtime_revision
    return value


def _support_model_snapshot_value(model: object) -> OrderedDict[str, object]:
    value: OrderedDict[str, object] = OrderedDict(
        (
            ("model", _field(model, "model")),
            (
                "provider_model_id_digest",
                _field(model, "provider_model_id_digest"),
            ),
            ("config_digest", _field(model, "config_digest")),
        )
    )
    runtime_revision = _optional_field(model, "runtime_revision")
    if runtime_revision is not None:
        value["runtime_revision"] = runtime_revision
    value["backend_topology_digest"] = _field(model, "backend_topology_digest")
    return value


def _routing_recipe_input_value(spec: object) -> OrderedDict[str, object]:
    return OrderedDict(
        (
            ("id", _field(spec, "id")),
            ("value_kind", _field(spec, "value_kind")),
        )
    )


def _routing_recipe_projection_value(spec: object) -> OrderedDict[str, object]:
    return OrderedDict(
        (
            ("id", _field(spec, "id")),
            ("value_kind", _field(spec, "value_kind")),
            ("outcome_binding", _field(spec, "outcome_binding")),
        )
    )


def _routing_recipe_plan_value(
    plan: object, *, canonical: bool
) -> OrderedDict[str, object]:
    arm_ids = list(_field(plan, "arm_ids"))
    signals = list(_field(plan, "signals"))
    projections = list(_field(plan, "projections"))
    top_k = list(_field(plan, "top_k"))
    if canonical:
        arm_ids.sort()
        signals.sort(key=lambda spec: _field(spec, "id"))
        projections.sort(key=lambda spec: _field(spec, "id"))
        top_k.sort()
    value: OrderedDict[str, object] = OrderedDict(
        (
            ("contract_version", _field(plan, "contract_version")),
            ("plan_digest", _field(plan, "plan_digest")),
            ("target_snapshot_digest", _field(plan, "target_snapshot_digest")),
            ("arm_ids", arm_ids),
        )
    )
    fallback_arm_id = _optional_field(plan, "fallback_arm_id")
    if fallback_arm_id:
        value["fallback_arm_id"] = fallback_arm_id
    value["signals"] = [_routing_recipe_input_value(spec) for spec in signals]
    value["projections"] = [
        _routing_recipe_projection_value(spec) for spec in projections
    ]
    value["top_k"] = top_k
    return value


def _routing_recipe_plan_digest_value(plan: object) -> OrderedDict[str, object]:
    arm_ids = sorted(_field(plan, "arm_ids"))
    signals = sorted(_field(plan, "signals"), key=lambda spec: _field(spec, "id"))
    projections = sorted(
        _field(plan, "projections"), key=lambda spec: _field(spec, "id")
    )
    return OrderedDict(
        (
            ("ContractVersion", _field(plan, "contract_version")),
            ("TargetSnapshotDigest", _field(plan, "target_snapshot_digest")),
            ("ArmIDs", list(arm_ids)),
            ("FallbackArmID", _optional_field(plan, "fallback_arm_id") or ""),
            (
                "Signals",
                [_routing_recipe_input_value(spec) for spec in signals],
            ),
            (
                "Projections",
                [_routing_recipe_projection_value(spec) for spec in projections],
            ),
            ("TopK", sorted(_field(plan, "top_k"))),
        )
    )


def _mixture_value(mixture: object) -> dict[str, object]:
    decisions = [
        {
            "name": _field(decision, "name"),
            "algorithm": _field(decision, "algorithm"),
            "arm_ids": list(_field(decision, "arm_ids")),
        }
        for decision in (_optional_field(mixture, "decisions") or ())
    ]
    value: dict[str, object] = {
        "schema_version": _field(mixture, "schema_version"),
        "id": _field(mixture, "id"),
        "entrypoint_model": _field(mixture, "entrypoint_model"),
        "aliases": list(_field(mixture, "aliases")),
        "recipe_name": _field(mixture, "recipe_name"),
        "recipe_description": _field(mixture, "recipe_description"),
        "recipe_digest": _field(mixture, "recipe_digest"),
        "pool_digest": _field(mixture, "pool_digest"),
        "selector_policy_digest": _field(mixture, "selector_policy_digest"),
        "selector_digest": _field(mixture, "selector_digest"),
        "adaptation_digest": _field(mixture, "adaptation_digest"),
        "binding_digest": _field(mixture, "binding_digest"),
        "model_arms": [_model_arm_value(arm) for arm in _field(mixture, "model_arms")],
        "support_models": [
            _support_model_value(model)
            for model in (_optional_field(mixture, "support_models") or ())
        ],
        "decisions": decisions,
        "routing_recipe_plan": _routing_recipe_plan_value(
            _field(mixture, "routing_recipe_plan"), canonical=True
        ),
    }
    fallback_arm_id = _optional_field(mixture, "fallback_arm_id")
    if fallback_arm_id is not None:
        value["fallback_arm_id"] = fallback_arm_id
    return value


def _mixture_snapshot_value(mixture: object) -> OrderedDict[str, object]:
    """Match Go's declared ``ManifestMixture`` order for an exact digest."""

    value: OrderedDict[str, object] = OrderedDict(
        (
            ("schema_version", _field(mixture, "schema_version")),
            ("id", _field(mixture, "id")),
            ("entrypoint_model", _field(mixture, "entrypoint_model")),
            ("aliases", list(_field(mixture, "aliases"))),
            ("recipe_name", _field(mixture, "recipe_name")),
            ("recipe_description", _field(mixture, "recipe_description")),
            ("recipe_digest", _field(mixture, "recipe_digest")),
            ("pool_digest", _field(mixture, "pool_digest")),
            (
                "selector_policy_digest",
                _field(mixture, "selector_policy_digest"),
            ),
            ("selector_digest", _field(mixture, "selector_digest")),
            ("adaptation_digest", _field(mixture, "adaptation_digest")),
            ("binding_digest", _field(mixture, "binding_digest")),
            (
                "model_arms",
                [
                    _model_arm_snapshot_value(arm)
                    for arm in _field(mixture, "model_arms")
                ],
            ),
            (
                "support_models",
                [
                    _support_model_snapshot_value(model)
                    for model in (_optional_field(mixture, "support_models") or ())
                ],
            ),
        )
    )
    fallback_arm_id = _optional_field(mixture, "fallback_arm_id")
    if fallback_arm_id:
        value["fallback_arm_id"] = fallback_arm_id
    value["decisions"] = [
        OrderedDict(
            (
                ("name", _field(decision, "name")),
                ("algorithm", _field(decision, "algorithm")),
                ("arm_ids", list(_field(decision, "arm_ids"))),
            )
        )
        for decision in (_optional_field(mixture, "decisions") or ())
    ]
    value["routing_recipe_plan"] = _routing_recipe_plan_value(
        _field(mixture, "routing_recipe_plan"), canonical=False
    )
    return value


def _capacity_slo_value(slo: object) -> OrderedDict[str, object]:
    """Match Go's declared CapacitySLO field order inside the semantic map."""

    return OrderedDict(
        (
            ("schema_version", _field(slo, "schema_version")),
            ("required_concurrency", _field(slo, "required_concurrency")),
            ("max_latency_p95_ms", _field(slo, "max_latency_p95_ms")),
            ("max_error_rate", _field(slo, "max_error_rate")),
            ("min_throughput_rps", _field(slo, "min_throughput_rps")),
            (
                "min_throughput_scaling_efficiency",
                _field(slo, "min_throughput_scaling_efficiency"),
            ),
        )
    )


def _capacity_load_protocol_value(
    protocol: object,
) -> OrderedDict[str, object]:
    """Match Go's declared CapacityLoadProtocol field order."""

    return OrderedDict(
        (
            ("schema_version", _field(protocol, "schema_version")),
            ("kind", _field(protocol, "kind")),
            (
                "concurrency_levels",
                list(_field(protocol, "concurrency_levels")),
            ),
            (
                "warmup_request_multiplier",
                _field(protocol, "warmup_request_multiplier"),
            ),
            (
                "measurement_requests_per_repetition",
                _field(protocol, "measurement_requests_per_repetition"),
            ),
            (
                "repetitions_per_level",
                _field(protocol, "repetitions_per_level"),
            ),
            (
                "minimum_measurement_clusters_per_level",
                _field(protocol, "minimum_measurement_clusters_per_level"),
            ),
            ("confidence_level", _field(protocol, "confidence_level")),
            (
                "max_error_rate_cluster_range",
                _field(protocol, "max_error_rate_cluster_range"),
            ),
            ("max_throughput_cv", _field(protocol, "max_throughput_cv")),
            ("max_latency_p95_cv", _field(protocol, "max_latency_p95_cv")),
        )
    )


def manifest_semantic_value(manifest: object) -> dict[str, object]:
    """Return the exact semantic value hashed by Go manifestSemanticDigest."""

    target_source = _field(manifest, "target")
    target: dict[str, object] = {
        "schema_version": _field(target_source, "schema_version"),
        "id": _field(target_source, "id"),
        "kind": _field(target_source, "kind"),
    }
    for name in (
        "router_api_url",
        "envoy_url",
        "router_api_key",
        "envoy_api_key",
        "backend_topology_digest",
    ):
        optional = _optional_field(target_source, name)
        if optional is not None:
            if name.endswith("_api_key"):
                # Go keeps SecretRef as a struct in the semantic map, so its
                # declaration order is significant even though map keys sort.
                optional = OrderedDict(
                    (
                        ("schema_version", _field(optional, "schema_version")),
                        ("env", _field(optional, "env")),
                    )
                )
            target[name] = optional
    for name in (
        "agent_task_ledger",
        "fault_recovery_ledger",
        "hard_policy_ledger",
        "production_experiment_ledger",
    ):
        endpoint = _optional_field(target_source, name)
        if endpoint is None:
            continue
        endpoint_value: OrderedDict[str, object] = OrderedDict(
            (
                ("schema_version", _field(endpoint, "schema_version")),
                ("url", _field(endpoint, "url")),
            )
        )
        api_key = _optional_field(endpoint, "api_key")
        if api_key is not None:
            endpoint_value["api_key"] = OrderedDict(
                (
                    ("schema_version", _field(api_key, "schema_version")),
                    ("env", _field(api_key, "env")),
                )
            )
        endpoint_value["timeout_seconds"] = _field(endpoint, "timeout_seconds")
        target[name] = endpoint_value
    mixture = _optional_field(target_source, "mixture")
    if mixture is not None:
        target["mixture"] = _mixture_value(mixture)

    value: dict[str, object] = {
        "schema_version": _field(manifest, "schema_version"),
        "run_id": _field(manifest, "run_id"),
        "name": _field(manifest, "name"),
        "description": _field(manifest, "description"),
        "mode": _field(manifest, "mode"),
        "target": target,
        "change_profile": _field(manifest, "change_profile"),
        "gate_contract_version": _field(manifest, "gate_contract_version"),
        "suite_ids": list(_field(manifest, "suite_ids")),
        "suite_revisions": dict(_field(manifest, "suite_revisions")),
        "suite_executors": dict(_field(manifest, "suite_executors")),
        "track_ids": list(_field(manifest, "track_ids")),
        "sample_limit": _field(manifest, "sample_limit"),
        "concurrency": _field(manifest, "concurrency"),
        "seed": _field(manifest, "seed"),
        "created_at": _go_timestamp(_field(manifest, "created_at")),
        "code_revision": _field(manifest, "code_revision"),
        "config_digest": _field(manifest, "config_digest"),
        "policy_snapshot_digest": _field(manifest, "policy_snapshot_digest"),
        "redaction_policy": _field(manifest, "redaction_policy"),
    }
    baseline_run_id = _optional_field(manifest, "baseline_run_id")
    if baseline_run_id is not None:
        value["baseline_run_id"] = baseline_run_id
    capacity_slo = _optional_field(manifest, "capacity_slo")
    if capacity_slo is not None:
        value["capacity_slo"] = _capacity_slo_value(capacity_slo)
    capacity_load_protocol = _optional_field(manifest, "capacity_load_protocol")
    if capacity_load_protocol is not None:
        value["capacity_load_protocol"] = _capacity_load_protocol_value(
            capacity_load_protocol
        )
    return value


def _go_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("canonical JSON does not support non-finite floats")
    absolute = abs(value)
    decimal = Decimal(repr(value))
    if absolute != 0 and (
        absolute < _GO_JSON_SCIENTIFIC_LOWER_BOUND
        or absolute >= _GO_JSON_SCIENTIFIC_UPPER_BOUND
    ):
        sign, digits, exponent = decimal.as_tuple()
        coefficient = "".join(str(digit) for digit in digits).rstrip("0") or "0"
        scientific_exponent = exponent + len(digits) - 1
        mantissa = coefficient[0]
        if len(coefficient) > 1:
            mantissa += "." + coefficient[1:]
        prefix = "-" if sign else ""
        exponent_sign = "+" if scientific_exponent >= 0 else "-"
        return f"{prefix}{mantissa}e{exponent_sign}{abs(scientific_exponent)}"
    encoded = format(decimal, "f")
    if "." in encoded:
        encoded = encoded.rstrip("0").rstrip(".")
    return encoded


def _go_json_scalar(value: object) -> str | None:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return (
            json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _go_float(value)
    return None


def _go_json(value: object) -> str:
    scalar = _go_json_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, OrderedDict):
        return (
            "{"
            + ",".join(
                f"{_go_json(str(key))}:{_go_json(item)}" for key, item in value.items()
            )
            + "}"
        )
    if isinstance(value, Mapping):
        return (
            "{"
            + ",".join(
                f"{_go_json(str(key))}:{_go_json(value[key])}" for key in sorted(value)
            )
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_go_json(item) for item in value) + "]"
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def mixture_target_id(recipe_name: str) -> str:
    """Return the exact server-owned target identity for one Recipe name."""

    return "mom-" + sha256(recipe_name.encode("utf-8")).hexdigest()


def model_pool_snapshot_digest(arms: object) -> str:
    """Bind a frozen, model-ordered arm list using the Go snapshot contract."""

    value = OrderedDict(
        (("model_arms", [_model_arm_snapshot_value(arm) for arm in arms]),)
    )
    return "sha256:" + sha256(_go_json(value).encode("utf-8")).hexdigest()


def selector_snapshot_digest(policy_digest: str, models: object) -> str:
    """Bind selector policy and executable support-model identities."""

    value = OrderedDict(
        (
            ("policy_digest", policy_digest),
            (
                "support_models",
                [_support_model_snapshot_value(model) for model in models],
            ),
        )
    )
    return "sha256:" + sha256(_go_json(value).encode("utf-8")).hexdigest()


def routing_recipe_target_snapshot_digest(mixture: object) -> str:
    """Bind the six immutable Mixture components without hashing the plan itself."""

    value = {
        "adaptation_digest": _field(mixture, "adaptation_digest"),
        "binding_digest": _field(mixture, "binding_digest"),
        "pool_digest": _field(mixture, "pool_digest"),
        "recipe_digest": _field(mixture, "recipe_digest"),
        "selector_digest": _field(mixture, "selector_digest"),
        "selector_policy_digest": _field(mixture, "selector_policy_digest"),
    }
    return "sha256:" + sha256(_go_json(value).encode("utf-8")).hexdigest()


def routing_recipe_plan_digest(plan: object) -> str:
    """Return the exact Go identity of a canonical routing Recipe plan body."""

    value = _routing_recipe_plan_digest_value(plan)
    return "sha256:" + sha256(_go_json(value).encode("utf-8")).hexdigest()


def mixture_snapshot_digest(mixture: object) -> str:
    """Digest every field of the server-sealed Mixture exactly as Go does."""

    return (
        "sha256:"
        + sha256(_go_json(_mixture_snapshot_value(mixture)).encode("utf-8")).hexdigest()
    )


def manifest_semantic_digest(manifest: object) -> str:
    encoded = _go_json(manifest_semantic_value(manifest)).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def require_manifest_digest(manifest: object) -> None:
    expected = manifest_semantic_digest(manifest)
    if _field(manifest, "manifest_digest") != expected:
        raise ValueError(
            "manifest_digest does not match the immutable manifest semantic value"
        )


def seal_manifest_fields(fields: Mapping[str, object]) -> dict[str, object]:
    """Create the one valid manifest payload from explicit semantic fields."""

    sealed = dict(fields)
    sealed["manifest_digest"] = manifest_semantic_digest(sealed)
    return sealed
