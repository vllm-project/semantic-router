"""Migration of retired embedding models to Vela Embedding.

The model runtime serves Vela Embedding (``mmbert``), Qwen3-Embedding
(``qwen3``) and Vela Omni (``multimodal``). EmbeddingGemma (``gemma``) and
MiniLM (``bert``) are retired, so every place that selected them, explicitly
or through the former MiniLM default, moves to Vela Embedding. Vectors of
different models are not comparable: a move that strands stored vectors is an
action note.
"""

from __future__ import annotations

from typing import Any

from cli.config_migration_legacy_models import replacement_for
from cli.config_migration_notes import MigrationNotes
from cli.config_migration_paths import as_list, decisions, dict_at
from cli.model_runtime_retired import REMOVED_EMBEDDING_BACKENDS

RETIRED_EMBEDDING_TYPES = frozenset({"gemma", "bert"})
VELA_EMBEDDING = "mmbert"
VELA_EMBEDDING_PATH = "models/Vela-1.0-Encoder-307M-Embedding"

_SEMANTIC = "global.model_catalog.embeddings.semantic"
_RETIRED_PATH_SLOTS = ("gemma_model_path", "bert_model_path")
_MODEL_PATH_SLOTS = ("qwen3_model_path", "mmbert_model_path", "multimodal_model_path")
# Store blocks whose unset embedding_model fell back to a semantic model path,
# in the router's fallback order, and the field naming each store's backend.
_FALLBACK_ORDER = (
    ("mmbert_model_path", "mmbert"),
    ("multimodal_model_path", "multimodal"),
    ("qwen3_model_path", "qwen3"),
    ("gemma_model_path", "gemma"),
)
_STORE_BACKEND_FIELDS = {
    "response_cache": "backend_type",
    "memory": "backend",
    "vector_store": "backend_type",
}
# RAG backends that embed the query with the router's default embedding model.
_RAG_EMBEDDING_BACKENDS = frozenset({"milvus", "qdrant", "hybrid"})


def migrate_embedding_models(canonical: dict[str, Any], notes: MigrationNotes) -> None:
    semantic = dict_at(canonical, "global", "model_catalog", "embeddings", "semantic")
    fallback = _store_fallback(semantic or {})
    if semantic is not None:
        _migrate_semantic(semantic, notes)
    _migrate_stores(canonical, fallback, notes)
    _migrate_model_selection(canonical, notes)
    for path, decision in decisions(canonical):
        _note_rag_collections(path, decision, notes)


def _store_fallback(semantic: dict[str, Any]) -> str:
    for slot, model in _FALLBACK_ORDER:
        if semantic.get(slot):
            return model
    return "bert"


def _migrate_semantic(semantic: dict[str, Any], notes: MigrationNotes) -> None:
    config = semantic.get("embedding_config")
    if isinstance(config, dict):
        backend = str(config.get("backend") or "").strip().lower()
        if backend in REMOVED_EMBEDDING_BACKENDS:
            config.pop("backend")
            notes.changed(
                _SEMANTIC + ".embedding_config.backend",
                f"removed {backend}; the model runtime serves local embeddings",
            )
        model_type = str(config.get("model_type") or "").strip().lower()
        if model_type in RETIRED_EMBEDDING_TYPES:
            config["model_type"] = VELA_EMBEDDING
            semantic.setdefault("mmbert_model_path", VELA_EMBEDDING_PATH)
            notes.changed(
                _SEMANTIC + ".embedding_config.model_type",
                f"{model_type} -> mmbert (Vela Embedding); stored vectors must be "
                "re-embedded and similarity thresholds re-checked",
            )
    for slot in _RETIRED_PATH_SLOTS:
        if slot in semantic:
            semantic.pop(slot)
            notes.changed(
                f"{_SEMANTIC}.{slot}",
                "removed; the runtime has no EmbeddingGemma or MiniLM family "
                "(use Vela Embedding or an OpenAI-compatible endpoint)",
            )
    for slot in _MODEL_PATH_SLOTS:
        legacy = semantic.get(slot)
        replacement = replacement_for(legacy)
        if replacement is not None:
            semantic[slot] = replacement.target
            notes.changed(
                f"{_SEMANTIC}.{slot}",
                f"{legacy} -> {replacement.target}: {replacement.note}",
            )


def _migrate_stores(
    canonical: dict[str, Any], fallback: str, notes: MigrationNotes
) -> None:
    stores = dict_at(canonical, "global", "stores")
    if stores is None:
        return
    for name, store in stores.items():
        if not isinstance(store, dict):
            continue
        configured = str(store.get("embedding_model") or "").strip().lower()
        if configured:
            previous = configured
        elif name in _STORE_BACKEND_FIELDS and store.get("enabled"):
            previous = "bert" if name == "vector_store" else fallback
        else:
            continue
        if previous not in RETIRED_EMBEDDING_TYPES:
            continue
        store["embedding_model"] = VELA_EMBEDDING
        path = f"global.stores.{name}.embedding_model"
        change = f"{previous} -> mmbert (Vela Embedding)"
        if not configured:
            change += f"; {previous} was this store's default"
        backend = str(store.get(_STORE_BACKEND_FIELDS.get(name, "backend")) or "")
        if backend.strip().lower() in {"", "memory"} and name == "response_cache":
            notes.changed(
                path,
                change + "; in-memory entries refill with new traffic, "
                "re-check similarity_threshold",
            )
        else:
            notes.action(
                path,
                change + "; re-embed the stored vectors with Vela Embedding "
                "and re-check similarity thresholds",
            )


def _migrate_model_selection(canonical: dict[str, Any], notes: MigrationNotes) -> None:
    ml = dict_at(canonical, "global", "router", "model_selection", "ml")
    if ml is None:
        return
    model_type = str(ml.get("model_type") or "").strip().lower()
    if model_type not in RETIRED_EMBEDDING_TYPES:
        return
    ml["model_type"] = VELA_EMBEDDING
    notes.action(
        "global.router.model_selection.ml.model_type",
        f"{model_type} -> mmbert (Vela Embedding); retrain the selection models "
        "on Vela Embedding vectors, the published ones match their embedding model",
    )


def _note_rag_collections(
    path: str, decision: dict[str, Any], notes: MigrationNotes
) -> None:
    for index, plugin in enumerate(as_list(decision.get("plugins"))):
        if not isinstance(plugin, dict) or plugin.get("type") != "rag":
            continue
        configuration = dict_at(plugin, "configuration", default={})
        backend = str(configuration.get("backend") or "").strip().lower()
        backend_config = dict_at(configuration, "backend_config", default={})
        embeds_query = backend in _RAG_EMBEDDING_BACKENDS or (
            backend == "external_api"
            and "embedding" in str(backend_config.get("request_format") or "")
        )
        if not embeds_query:
            continue
        collection = backend_config.get("collection")
        target = f"collection {collection!r}" if collection else "the collection"
        notes.action(
            f"{path}.plugins[{index}].configuration.backend",
            "queries are now embedded with Vela Embedding (768 dimensions) instead "
            f"of MiniLM (384); re-embed {target} with Vela Embedding",
        )
