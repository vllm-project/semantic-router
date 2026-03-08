"""Typed runtime-misc compatibility blocks for the TD001 CLI migration."""

from pydantic import BaseModel, ConfigDict

from cli.compat_blocks_semantic_cache import SemanticCacheMilvusCompatConfig


class RuntimeTopLevelCompatConfig(BaseModel):
    """Typed schema for narrow top-level runtime keys still outside UserConfig."""

    model_config = ConfigDict(extra="forbid")

    config_source: str | None = None
    mom_registry: dict[str, str] | None = None
    strategy: str | None = None


class VectorStoreMemoryCompatConfig(BaseModel):
    """Typed schema for vector_store.memory."""

    model_config = ConfigDict(extra="forbid")

    max_entries_per_store: int | None = None


class VectorStoreLlamaStackCompatConfig(BaseModel):
    """Typed schema for vector_store.llama_stack."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str | None = None
    auth_token: str | None = None
    embedding_model: str | None = None
    request_timeout_seconds: int | None = None
    search_type: str | None = None


class VectorStoreCompatConfig(BaseModel):
    """Typed schema for vector_store."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    backend_type: str | None = None
    file_storage_dir: str | None = None
    max_file_size_mb: int | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    ingestion_workers: int | None = None
    supported_formats: list[str] | None = None
    milvus: SemanticCacheMilvusCompatConfig | None = None
    memory: VectorStoreMemoryCompatConfig | None = None
    llama_stack: VectorStoreLlamaStackCompatConfig | None = None
