"""Model execution fields from before the built-in model runtime served every
local model: the candle, ort and openvino providers, their precision and
kernel fields, deployment profiles, retired Omni bundles and the NLI
explainer with the fields that asked it.

The lines mirror src/semantic-router/pkg/config/config_forms_model_runtime.go.
"""

from __future__ import annotations

from typing import Any

from cli.config_recipe_scopes import recipe_scopes

MIGRATE = "run `vllm-sr config migrate`"
REMOVED_MODEL_PROVIDERS = ("candle", "ort", "openvino")
REMOVED_EXECUTION_FIELDS = ("precision", "custom_ops_profile", "compilation_cache_dir")
RETIRED_OMNI_BUNDLES = {
    "/opt/router-model-artifacts/vela-1.0-omni-nano": "vllm-sr/Vela-1.0-Omni-Nano",
    "/opt/router-model-artifacts/vela-1.0-omni-mini": "vllm-sr/Vela-1.0-Omni-Mini",
}
RETIRED_NLI_ALIASES = frozenset(
    key
    for alias in (
        "models/mom-halugate-explainer",
        "hallucination-explainer",
        "halugate-explainer",
        "nli-explainer",
        "tasksource/ModernBERT-base-nli",
    )
    for key in ((alias,) if "/" in alias else (alias, f"models/{alias}"))
)


def removed_model_runtime_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed model execution fields."""

    lines: list[str] = []
    deployments = _mapping_at(data, "global", "model_catalog", "deployments")
    for name, deployment in deployments.items():
        if not isinstance(deployment, dict):
            continue
        path = f"global.model_catalog.deployments.{name}"
        artifact = deployment.get("artifact")
        if (
            isinstance(artifact, str)
            and artifact.strip().rstrip("/") in RETIRED_NLI_ALIASES
        ):
            lines.append(
                _line(
                    path,
                    "removed_field",
                    "the deployment of the retired NLI explainer is removed",
                    "drop it; nothing explains a hallucination with an NLI model",
                )
            )
        provider = deployment.get("provider")
        if (
            isinstance(provider, str)
            and provider.strip().lower() in REMOVED_MODEL_PROVIDERS
        ):
            lines.append(
                _line(
                    path,
                    "invalid_value",
                    f"provider {provider} is removed: the built-in model runtime "
                    "serves every local model",
                    "provider: model_runtime, on its device, with its artifact's "
                    "Hub repository",
                )
            )
        for field in REMOVED_EXECUTION_FIELDS:
            if field in deployment:
                lines.append(
                    _line(
                        f"{path}.{field}",
                        "removed_field",
                        f"{field} is removed",
                        "nothing; the model runtime selects numerics and kernels, "
                        "and dtype sets a precision",
                    )
                )
        if "profile" in deployment:
            lines.append(
                _line(
                    f"{path}.profile",
                    "removed_field",
                    "profile is removed",
                    "the engine arguments it stood for: max_num_seqs, "
                    "max_num_batched_tokens, batch_window_ms, "
                    "enable_shared_context, dtype",
                )
            )
        bundle = (
            RETIRED_OMNI_BUNDLES.get(artifact.rstrip("/"))
            if isinstance(artifact, str)
            else None
        )
        if bundle:
            lines.append(
                _line(
                    f"{path}.artifact",
                    "invalid_value",
                    f"artifact {artifact} is an Omni bundle router images no "
                    "longer ship",
                    f"artifact: {bundle}, the published repository the runtime serves",
                )
            )
    for prefix, recipe in recipe_scopes(data):
        rules = _mapping_at(recipe, "signals").get("hallucination")
        for index, rule in enumerate(_items(rules)):
            if "use_nli" in rule:
                lines.append(
                    _line(
                        f"{prefix}.signals.hallucination[{_label(rule, index)}].use_nli",
                        "removed_field",
                        "use_nli is removed: NLI explanations are retired",
                        "nothing; the rule's spans stay in the response",
                    )
                )
        for index, decision in enumerate(_items(recipe.get("decisions"))):
            grounding = _mapping_at(decision, "algorithm", "fusion", "grounding")
            if "nli_contradiction_penalty" in grounding:
                lines.append(
                    _line(
                        f"{prefix}.decisions[{_label(decision, index)}]"
                        ".algorithm.fusion.grounding.nli_contradiction_penalty",
                        "removed_field",
                        "nli_contradiction_penalty is removed",
                        "contradiction_penalty; grounding reads the hallucination "
                        "signal, not an NLI model",
                    )
                )
    if "polarity_guard" in _mapping_at(data, "global", "stores", "response_cache"):
        lines.append(
            _line(
                "global.stores.response_cache.polarity_guard",
                "removed_field",
                "polarity_guard is removed: the NLI tier is retired",
                "nothing; the lexical polarity guard always runs",
            )
        )
    return lines


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
