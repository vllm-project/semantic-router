"""Context forms the Router no longer reads: the rag, memory and
context_compression plugins (one context plugin per decision), their recall
and scoring fields (the sources' and compress's match blocks), RAG's
min_confidence_threshold (a fact_check condition in the decision's rules),
and the rag.reranker binding and the rerank provider (the decision model
judges RAG relevance).

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites it; the CLI refuses them with the same lines.
"""

from __future__ import annotations

from typing import Any

MIGRATE = "run `vllm-sr config migrate`"

RAG_RECALL_FIELDS = ("top_k", "similarity_threshold", "rerank")
MEMORY_RECALL_FIELDS = ("retrieval_limit", "similarity_threshold")
RECALL_FIELD_TARGETS = {
    "top_k": "match.recall.top_k",
    "retrieval_limit": "match.recall.top_k",
    "similarity_threshold": "match.recall.min_similarity",
    "rerank": "match.judge with the relevance preset on the decision model",
}
REMOVED_PLUGIN_TYPES = {
    "rag": "a rag source of the decision's context plugin",
    "memory": "a memory source of the decision's context plugin",
    "context_compression": "the compress block and budget of the decision's context plugin",
}


def removed_context_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed context forms in data, in the
    order the Router reports them."""

    field_lines: list[str] = []
    type_lines: list[str] = []
    for prefix, routing in _routing_scopes(data):
        decisions = routing.get("decisions")
        for index, decision in enumerate(
            decisions if isinstance(decisions, list) else []
        ):
            if not isinstance(decision, dict):
                continue
            name = decision.get("name")
            label = name if isinstance(name, str) and name else str(index)
            plugins = decision.get("plugins")
            for position, plugin in enumerate(
                plugins if isinstance(plugins, list) else []
            ):
                if not isinstance(plugin, dict):
                    continue
                path = f"{prefix}.decisions[{label}].plugins[{position}]"
                for code, what, becomes in _field_forms(plugin):
                    field_lines.append(f"{path}: {code}: {what}; {MIGRATE} ({becomes})")
                plugin_type = plugin.get("type")
                if plugin_type in REMOVED_PLUGIN_TYPES:
                    type_lines.append(
                        f"{path}: removed_plugin_type: plugin {plugin_type} is removed; "
                        f"{MIGRATE} ({REMOVED_PLUGIN_TYPES[plugin_type]})"
                    )
    return field_lines + type_lines + _model_catalog_forms(data)


def _field_forms(plugin: dict[str, Any]) -> list[tuple[str, str, str]]:
    configuration = plugin.get("configuration")
    if not isinstance(configuration, dict):
        return []
    plugin_type = plugin.get("type")
    forms: list[tuple[str, str, str]] = []
    if plugin_type == "rag" and "min_confidence_threshold" in configuration:
        forms.append(
            (
                "removed_field",
                "rag min_confidence_threshold is removed: it retrieved only when "
                "a fact_check rule matched",
                "a fact_check condition in the decision's rules",
            )
        )
    recall_fields = {"rag": RAG_RECALL_FIELDS, "memory": MEMORY_RECALL_FIELDS}.get(
        plugin_type, ()
    )
    written = [field for field in recall_fields if field in configuration]
    if written:
        verb = "is" if len(written) == 1 else "are"
        forms.append(
            (
                "removed_field",
                f"{plugin_type} {', '.join(written)} {verb} removed",
                ", ".join(RECALL_FIELD_TARGETS[field] for field in written),
            )
        )
    if plugin_type == "context_compression" and "scoring" in configuration:
        forms.append(
            (
                "removed_field",
                "context_compression scoring is removed: segments are judged, not "
                "embedded",
                "compress.match: the relevance judge by default, or recall: "
                "{method: bm25} for lexical order",
            )
        )
    return forms


def _model_catalog_forms(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    global_section = data.get("global")
    catalog = (
        global_section.get("model_catalog")
        if isinstance(global_section, dict)
        else None
    )
    catalog = catalog if isinstance(catalog, dict) else {}
    binding_maps = [("global.model_catalog.bindings", catalog.get("bindings"))]
    binding_maps += [
        (f"{prefix}.model_bindings", routing.get("model_bindings"))
        for prefix, routing in _routing_scopes(data)
    ]
    for prefix, bindings in binding_maps:
        if isinstance(bindings, dict) and "rag.reranker" in bindings:
            lines.append(
                f"{prefix}.rag.reranker: removed_field: the rag.reranker binding is "
                f"removed: the decision model judges RAG relevance; {MIGRATE} (the "
                "rag source's match.judge)"
            )
    deployments = catalog.get("deployments")
    for name, deployment in (
        deployments.items() if isinstance(deployments, dict) else []
    ):
        if isinstance(deployment, dict) and deployment.get("provider") == "rerank":
            lines.append(
                f"global.model_catalog.deployments.{name}: invalid_value: provider "
                "rerank is removed: the decision model judges RAG relevance; "
                f"{MIGRATE} (the rag source's match.judge; the deployment is deleted)"
            )
    return lines


def _routing_scopes(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    scopes: list[tuple[str, dict[str, Any]]] = []
    routing = data.get("routing")
    if isinstance(routing, dict):
        scopes.append(("routing", routing))
    recipes = data.get("recipes")
    for index, recipe in enumerate(recipes if isinstance(recipes, list) else []):
        if not isinstance(recipe, dict) or not isinstance(recipe.get("routing"), dict):
            continue
        name = recipe.get("name")
        label = name if isinstance(name, str) and name else str(index)
        scopes.append((f"recipes[{label}].routing", recipe["routing"]))
    return scopes
