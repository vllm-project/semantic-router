"""Memory-related configuration models for vLLM Semantic Router."""

from pydantic import BaseModel


class MemoryMilvusConfig(BaseModel):
    """Milvus configuration for memory storage."""

    address: str
    collection: str = "agentic_memory"
    dimension: int = 384


class MemoryValkeyConfig(BaseModel):
    """Valkey configuration for memory storage using the Search module."""

    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str | None = None
    timeout: int = 10
    collection_prefix: str = "mem:"
    index_name: str = "mem_idx"
    dimension: int = 384
    metric_type: str = "COSINE"
    index_m: int = 16
    index_ef_construction: int = 256
    tls_enabled: bool = False
    tls_ca_path: str | None = None
    tls_insecure_skip_verify: bool = False


class MemoryRedisCacheConfig(BaseModel):
    """Redis hot-cache configuration for memory retrieval."""

    enabled: bool = False
    address: str = "localhost:6379"
    ttl_seconds: int = 300
    db: int = 0
    key_prefix: str = "memory_cache:"
    password: str | None = ""


class MemoryConfig(BaseModel):
    """Agentic Memory configuration for cross-session memory.

    Query rewriting and fact extraction are enabled by adding external_models
    with role="memory_rewrite" or role="memory_extraction".
    See external_models configuration in providers section for details.

    The embedding_model is auto-detected from embedding_models if not specified.
    Priority: bert > mmbert > multimodal > qwen3 > gemma
    """

    enabled: bool = True
    backend: str = ""  # "" or "milvus" → Milvus (default); "valkey" → Valkey
    auto_store: bool = False  # Auto-store extracted facts after each response
    milvus: MemoryMilvusConfig | None = None
    valkey: MemoryValkeyConfig | None = None
    redis_cache: MemoryRedisCacheConfig | None = None
    # Embedding model to use for memory vectors
    # Options: "bert", "mmbert", "multimodal", "qwen3", "gemma"
    # If not set, auto-detected from embedding_models section (bert preferred)
    embedding_model: str | None = None
    default_retrieval_limit: int = 5
    default_similarity_threshold: float = 0.70
    extraction_batch_size: int = 10  # Extract every N turns
