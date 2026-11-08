"""Embedding settings the Router no longer reads: embedding bindings, the
stores' embedding model keywords and the vector store's width, admission by
embedding model keyword, and embeddings.semantic. Every store and collection
now names its embedding deployment, or embeds with embeddings.default.

The Router refuses each one with a line that names `vllm-sr config migrate`,
which rewrites it; the CLI refuses them with the same lines.
"""

from __future__ import annotations

from typing import Any

from cli.config_recipe_scopes import recipe_scopes

MIGRATE = "run `vllm-sr config migrate`"

KEYWORD_STORES = ("response_cache", "memory", "vector_store")
BINDING = (
    "the embedding binding is removed",
    "global.model_catalog.embeddings.default; a RAG source's "
    "backend_config.embedding.deployment for a routing scope's own",
)
KEYWORD = (
    "embedding_model is removed",
    "embedding.deployment names the store's deployment; without it the store "
    "embeds with embeddings.default",
)
WIDTH = ("embedding_dimension is removed", "embedding.dimension")
ADMISSION = (
    "admission by embedding model keyword is removed",
    "the admission of the embedding deployment, keyed by its name",
)
SEMANTIC = (
    "embeddings.semantic is removed",
    "embeddings.default names the embedding deployment; a store's embedding "
    "block names its own",
)


def removed_embedding_forms(data: dict[str, Any]) -> list[str]:
    """The one-line load errors of the removed embedding forms in data."""

    lines: list[str] = []
    for prefix, routing in recipe_scopes(data):
        bindings = routing.get("model_bindings")
        if isinstance(bindings, dict) and "embedding" in bindings:
            lines.append(_line(f"{prefix}.model_bindings.embedding", BINDING))
    global_section = data.get("global")
    if not isinstance(global_section, dict):
        return lines
    catalog = global_section.get("model_catalog")
    catalog = catalog if isinstance(catalog, dict) else {}
    bindings = catalog.get("bindings")
    if isinstance(bindings, dict) and "embedding" in bindings:
        lines.append(_line("global.model_catalog.bindings.embedding", BINDING))
    stores = global_section.get("stores")
    stores = stores if isinstance(stores, dict) else {}
    for name in KEYWORD_STORES:
        store = stores.get(name)
        if isinstance(store, dict) and "embedding_model" in store:
            lines.append(_line(f"global.stores.{name}.embedding_model", KEYWORD))
    vector_store = stores.get("vector_store")
    if isinstance(vector_store, dict) and "embedding_dimension" in vector_store:
        lines.append(_line("global.stores.vector_store.embedding_dimension", WIDTH))
    admission = catalog.get("admission")
    for key in admission if isinstance(admission, dict) else {}:
        if isinstance(key, str) and key.startswith("embedding:"):
            lines.append(_line(f"global.model_catalog.admission.{key}", ADMISSION))
    embeddings = catalog.get("embeddings")
    if isinstance(embeddings, dict) and "semantic" in embeddings:
        lines.append(_line("global.model_catalog.embeddings.semantic", SEMANTIC))
    return lines


def _line(path: str, form: tuple[str, str]) -> str:
    what, becomes = form
    return f"{path}: removed_field: {what}; {MIGRATE} ({becomes})"
