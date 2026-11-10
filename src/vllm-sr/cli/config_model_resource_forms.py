"""Ways to choose a model the Router read as deployments and bindings before
every model was a named deployment: a deployment's question_types and
served_name, deployments of the removed classifier providers, the
external[] models, binding objects and bindings for one rule, and a request
backend's connection fields.

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites it; the CLI refuses them with the same lines. This mirrors
src/semantic-router/pkg/config/config_forms_model_resources.go.
"""

from __future__ import annotations

from typing import Any

from cli.config_application_forms import (
    LABEL_DISTRIBUTION_CONTRACT,
    REMOVED_BINDING_CONSUMERS,
)
from cli.config_recipe_scopes import recipe_scopes

MIGRATE = "run `vllm-sr config migrate`"
REMOVED_CLASSIFIER_PROVIDERS = ("classify", "http")
LEGACY_BINDING_FIELDS = (
    "contract",
    "adapter",
    "head",
    "mapping_path",
    "operating_point",
)
CLASSIFIER_CONTRACTS = (
    "label_distribution.v1",
    "label_scores.v1",
    "label_decision.v1",
    "score.v1",
    "token_spans.v1",
)
LEGACY_CONNECTION_FIELDS = (
    "base_url",
    "api_key_env",
    "auth_header",
    "auth_prefix",
    "extra_headers",
)


def removed_model_resource_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed model-resource forms in data."""

    lines: list[str] = []
    catalog = _mapping_at(data, "global", "model_catalog")
    deployments = _mapping_at(catalog, "deployments")
    for name, deployment in deployments.items():
        if not isinstance(deployment, dict):
            continue
        path = f"global.model_catalog.deployments.{name}"
        if "question_types" in deployment:
            lines.append(
                _line(
                    f"{path}.question_types",
                    "removed_field",
                    "question_types is removed",
                    "capabilities.question_types",
                )
            )
        served = [f"{path}.served_name"] if "served_name" in deployment else []
        for index, replica in enumerate(_items(deployment.get("replicas"))):
            if "served_name" in replica:
                served.append(f"{path}.replicas[{_label(replica, index)}].served_name")
        for served_path in served:
            lines.append(
                _line(served_path, "removed_field", "served_name is removed", "model")
            )
        if "operating_point" in deployment:
            lines.append(
                _line(
                    f"{path}.operating_point",
                    "removed_field",
                    "a deployment's operating_point is removed: nothing reads it",
                    "nothing: migrate drops it; the model runtime applies the "
                    "operating point its package declares",
                )
            )
        provider = deployment.get("provider")
        if provider in REMOVED_CLASSIFIER_PROVIDERS:
            lines.append(
                _line(
                    path,
                    "invalid_value",
                    f"provider {provider} is removed: no signal asks a classifier service",
                    "drop the deployment; the decision model answers the signals",
                )
            )
    for prefix, bindings in _binding_maps(data):
        for key, binding in bindings.items():
            if not _signal_binding(str(key), binding):
                continue
            path = f"{prefix}.{key}"
            contract = binding.get("contract") if isinstance(binding, dict) else None
            if contract in CLASSIFIER_CONTRACTS:
                lines.append(
                    _line(
                        path,
                        "removed_field",
                        f"binding {key} asks a classifier head ({contract}), which no "
                        "signal asks",
                        "drop it; the decision model answers the signal",
                    )
                )
                continue
            fields = [
                field
                for field in LEGACY_BINDING_FIELDS
                if isinstance(binding, dict) and field in binding
            ]
            if fields:
                lines.append(
                    _line(
                        path,
                        "removed_field",
                        _removed_fields(path, fields),
                        f"{key}: {binding.get('deployment', '')}; the signal "
                        "decides what the deployment answers",
                    )
                )
            collection, _, rule = str(key).partition(".")
            if rule and collection in ("safety", "complexity"):
                lines.append(
                    _line(
                        path,
                        "removed_field",
                        "a binding for one rule is removed",
                        f"the rule's deployment: recipes[].signals.{collection}[{rule}].deployment",
                    )
                )
    for index, external in enumerate(_items(catalog.get("external"))):
        lines.append(
            _line(
                f"global.model_catalog.external[{_label(external, index)}]",
                "removed_section",
                "external models are removed",
                "drop it; the model_role: memory_rewrite model is an openai_chat "
                "deployment global.stores.memory.query_rewrite names",
            )
        )
    providers = data.get("providers")
    models = providers.get("models") if isinstance(providers, dict) else None
    for model_index, model in enumerate(_items(models)):
        model_label = _label(model, model_index)
        for index, backend in enumerate(_items(model.get("backend_refs"))):
            fields = [field for field in LEGACY_CONNECTION_FIELDS if field in backend]
            if not fields:
                continue
            path = f"providers.models[{model_label}].backend_refs[{_label(backend, index)}]"
            lines.append(
                _line(
                    path,
                    "removed_field",
                    _removed_fields(path, fields),
                    "endpoint (a full URL), auth {header, prefix, api_key_env} and "
                    "headers, as a deployment connects",
                )
            )
    return lines


def _binding_maps(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    maps = [
        (f"{prefix}.model_bindings", _mapping_at(recipe, "model_bindings"))
        for prefix, recipe in recipe_scopes(data)
    ]
    maps.append(
        (
            "global.model_catalog.bindings",
            _mapping_at(data, "global", "model_catalog", "bindings"),
        )
    )
    return maps


def _signal_binding(key: str, binding: Any) -> bool:
    """A binding of a signal the decision model answers, which the
    application and embedding forms leave in place."""

    if key == "embedding" or key in REMOVED_BINDING_CONSUMERS:
        return False
    if key.startswith("classifier."):
        return False
    contract = binding.get("contract") if isinstance(binding, dict) else None
    complexity = key == "complexity" or key.startswith("complexity.")
    return not (complexity and contract == LABEL_DISTRIBUTION_CONTRACT)


def _removed_fields(path: str, fields: list[str]) -> str:
    listed = fields[0]
    if len(fields) > 1:
        listed = ", ".join(fields[:-1]) + " and " + fields[-1]
    verb = "is" if len(fields) == 1 else "are"
    return f"{path}.{listed} {verb} removed"


def _line(path: str, code: str, what: str, becomes: str) -> str:
    return f"{path}: {code}: {what}; {MIGRATE} ({becomes})"


def _items(value: Any) -> list[dict[str, Any]]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _label(item: dict[str, Any], index: int) -> str:
    name = item.get("name")
    return name if isinstance(name, str) and name else str(index)


def _mapping_at(node: Any, *keys: str) -> dict[str, Any]:
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return node if isinstance(node, dict) else {}
