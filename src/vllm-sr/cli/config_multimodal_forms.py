"""Multimodal forms the Router no longer reads: a model's modality keyword
(ar, diffusion, omni), a decision whose output followed the modality of its
models instead of an output contract, and the image_gen plugin.

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites it where it can; the CLI refuses them with the same lines.
"""

from __future__ import annotations

from typing import Any

from cli.config_recipe_scopes import recipe_scopes

MIGRATE = "run `vllm-sr config migrate`"

LEGACY_MODALITY_LABELS = {"AR": "text", "DIFFUSION": "image", "BOTH": "text_image"}

IMAGE_GEN_BECOMES = (
    "an image model in providers.models (api_format: images, capabilities: "
    "[image_generation]) that the decision references, with output: "
    "{kind: image, action: generate}"
)

DECISION_OUTPUT_WHAT = (
    "a decision's output no longer follows the modality of its models"
)


def removed_multimodal_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed multimodal forms in data."""

    lines: list[str] = []
    models = _models_by_name(data)
    for prefix, recipe in recipe_scopes(data):
        labels = _modality_rule_labels(recipe)
        for path, decision in _named_items(
            f"{prefix}.decisions", recipe.get("decisions")
        ):
            becomes = _legacy_image_route(decision, labels, models)
            if becomes is not None:
                lines.append(
                    _line(path, "removed_field", DECISION_OUTPUT_WHAT, becomes)
                )
            plugins = decision.get("plugins")
            for index, plugin in enumerate(
                plugins if isinstance(plugins, list) else []
            ):
                if isinstance(plugin, dict) and plugin.get("type") == "image_gen":
                    lines.append(
                        _line(
                            f"{path}.plugins[{index}]",
                            "removed_plugin_type",
                            "plugin image_gen was removed",
                            IMAGE_GEN_BECOMES,
                        )
                    )
    for name, model in models.items():
        keyword = model.get("modality")
        if isinstance(keyword, str):
            what, becomes = _describe_modality_keyword(keyword)
            lines.append(
                _line(
                    f"providers.models[{name}].modality", "removed_field", what, becomes
                )
            )
    return lines


def _legacy_image_route(
    decision: dict[str, Any],
    labels: dict[str, list[str]],
    models: dict[str, dict[str, Any]],
) -> str | None:
    """What a decision without an output contract becomes when it took its
    output from the modality keywords of its models, or None."""

    if "output" in decision:
        return None
    referenced = [
        models.get(ref.get("model"), {})
        for ref in decision.get("modelRefs") or []
        if isinstance(ref, dict)
    ]
    keywords = [str(model.get("modality", "")).lower() for model in referenced]
    if not {"diffusion", "omni"} & set(keywords):
        return None
    read: list[str] = []
    _positive_modality_labels(decision.get("rules"), labels, False, read)
    text_and_image = "text_image" in read
    if not text_and_image and "diffusion" not in keywords and "image" not in read:
        return None
    rewritable = all(
        images and (text or not text_and_image)
        for text, images in (_model_writes(model) for model in referenced)
    )
    if text_and_image and rewritable:
        return "output: {kind: [text, image], action: generate}"
    if text_and_image:
        return (
            "nothing: migrate drops the decision, since a model it references cannot "
            "answer in text and generate images; reference models on api_format: "
            "responses that do both, with output: {kind: [text, image], action: "
            "generate}"
        )
    if rewritable:
        return "output: {kind: image, action: generate}"
    return (
        "nothing: migrate drops the decision, since a model it references generates "
        "no images; reference image models only, with output: {kind: image, action: "
        "generate}"
    )


def _model_writes(model: dict[str, Any]) -> tuple[bool, bool]:
    diffusion = str(model.get("modality", "")).lower() == "diffusion"
    capabilities = model.get("capabilities")
    images = diffusion or (
        isinstance(capabilities, list) and "image_generation" in capabilities
    )
    modalities = model.get("modalities")
    if isinstance(modalities, dict) and isinstance(modalities.get("output"), list):
        return "text" in modalities["output"], images
    return not diffusion, images


def _describe_modality_keyword(keyword: str) -> tuple[str, str]:
    lowered = keyword.lower()
    if lowered == "diffusion":
        return (
            "modality: diffusion is removed",
            "capabilities: [image_generation] and modalities: {input: [text], output: [image]}",
        )
    if lowered == "ar":
        return (
            "modality: ar is removed",
            "nothing: a model answers in text unless its modalities say otherwise",
        )
    if lowered == "omni":
        return (
            "modality: omni is removed",
            "modalities naming what the model reads and writes",
        )
    return f"modality: {keyword} is removed", "capabilities and modalities"


def _positive_modality_labels(
    node: Any, labels: dict[str, list[str]], negated: bool, read: list[str]
) -> None:
    if not isinstance(node, dict):
        return
    operator = node.get("operator")
    if isinstance(operator, str) and operator.upper() == "NOT":
        negated = not negated
    for child in node.get("conditions") or []:
        _positive_modality_labels(child, labels, negated, read)
    if negated or node.get("type") != "modality":
        return
    label = node.get("label")
    if isinstance(label, str) and label:
        read.append(label)
        return
    read.extend(labels.get(node.get("name"), []))


def _modality_rule_labels(recipe: dict[str, Any]) -> dict[str, list[str]]:
    signals = recipe.get("signals")
    rules = signals.get("modality") if isinstance(signals, dict) else None
    labels: dict[str, list[str]] = {}
    for rule in rules if isinstance(rules, list) else []:
        if not isinstance(rule, dict):
            continue
        name = rule.get("name")
        rule_labels = rule.get("labels")
        labels[name] = list(rule_labels) if isinstance(rule_labels, list) else []
        if not labels[name] and name in LEGACY_MODALITY_LABELS:
            labels[name] = [LEGACY_MODALITY_LABELS[name]]
    return labels


def _models_by_name(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    providers = data.get("providers")
    models = providers.get("models") if isinstance(providers, dict) else None
    return {
        model["name"]: model
        for model in (models if isinstance(models, list) else [])
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    }


def _named_items(prefix: str, items: Any) -> list[tuple[str, dict[str, Any]]]:
    named: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(items if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        label = name if isinstance(name, str) and name else str(index)
        named.append((f"{prefix}[{label}]", item))
    return named


def _line(path: str, code: str, what: str, becomes: str) -> str:
    return f"{path}: {code}: {what}; {MIGRATE} ({becomes})"
