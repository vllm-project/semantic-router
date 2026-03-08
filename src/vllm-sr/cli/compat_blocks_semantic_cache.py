"""Typed semantic_cache compatibility models for the TD001 CLI migration."""

from pydantic import BaseModel, ConfigDict, Field


class SemanticCacheRedisTLSCompatConfig(BaseModel):
    """Typed schema for semantic_cache.redis.connection.tls."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None


class SemanticCacheRedisConnectionCompatConfig(BaseModel):
    """Typed schema for semantic_cache.redis.connection."""

    model_config = ConfigDict(extra="forbid")

    host: str | None = None
    port: int | None = None
    database: int | None = None
    password: str | None = None
    timeout: int | None = None
    tls: SemanticCacheRedisTLSCompatConfig | None = None


class SemanticCacheRedisVectorFieldCompatConfig(BaseModel):
    """Typed schema for semantic_cache.redis.index.vector_field."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    dimension: int | None = None
    metric_type: str | None = None


class SemanticCacheRedisIndexParamsCompatConfig(BaseModel):
    """Typed schema for semantic_cache.redis.index.params."""

    model_config = ConfigDict(extra="forbid")

    M: int | None = None
    ef_construction: int | None = Field(default=None, alias="efConstruction")


class SemanticCacheRedisIndexCompatConfig(BaseModel):
    """Typed schema for semantic_cache.redis.index."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    prefix: str | None = None
    vector_field: SemanticCacheRedisVectorFieldCompatConfig | None = None
    index_type: str | None = None
    params: SemanticCacheRedisIndexParamsCompatConfig | None = None


class SemanticCacheRedisSearchCompatConfig(BaseModel):
    """Typed schema for semantic_cache.redis.search."""

    model_config = ConfigDict(extra="forbid")

    topk: int | None = None


class SemanticCacheRedisDevelopmentCompatConfig(BaseModel):
    """Typed schema for semantic_cache.redis.development."""

    model_config = ConfigDict(extra="forbid")

    drop_index_on_startup: bool | None = None
    auto_create_index: bool | None = None
    verbose_errors: bool | None = None


class SemanticCacheRedisLoggingCompatConfig(BaseModel):
    """Typed schema for semantic_cache.redis.logging."""

    model_config = ConfigDict(extra="forbid")

    level: str | None = None
    enable_query_log: bool | None = None
    enable_metrics: bool | None = None


class SemanticCacheRedisCompatConfig(BaseModel):
    """Typed schema for semantic_cache.redis."""

    model_config = ConfigDict(extra="forbid")

    connection: SemanticCacheRedisConnectionCompatConfig | None = None
    index: SemanticCacheRedisIndexCompatConfig | None = None
    search: SemanticCacheRedisSearchCompatConfig | None = None
    development: SemanticCacheRedisDevelopmentCompatConfig | None = None
    logging: SemanticCacheRedisLoggingCompatConfig | None = None


class SemanticCacheMilvusAuthCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.connection.auth."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    username: str | None = None
    password: str | None = None


class SemanticCacheMilvusTLSCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.connection.tls."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    cert_file: str | None = None
    key_file: str | None = None
    ca_file: str | None = None


class SemanticCacheMilvusConnectionCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.connection."""

    model_config = ConfigDict(extra="forbid")

    host: str | None = None
    port: int | None = None
    database: str | None = None
    timeout: int | None = None
    auth: SemanticCacheMilvusAuthCompatConfig | None = None
    tls: SemanticCacheMilvusTLSCompatConfig | None = None


class SemanticCacheMilvusVectorFieldCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.collection.vector_field."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    dimension: int | None = None
    metric_type: str | None = None


class SemanticCacheMilvusIndexParamsCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.collection.index.params."""

    model_config = ConfigDict(extra="forbid")

    M: int | None = None
    ef_construction: int | None = Field(default=None, alias="efConstruction")


class SemanticCacheMilvusIndexCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.collection.index."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = None
    params: SemanticCacheMilvusIndexParamsCompatConfig | None = None


class SemanticCacheMilvusCollectionCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.collection."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    vector_field: SemanticCacheMilvusVectorFieldCompatConfig | None = None
    index: SemanticCacheMilvusIndexCompatConfig | None = None


class SemanticCacheMilvusSearchParamsCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.search.params."""

    model_config = ConfigDict(extra="forbid")

    ef: int | None = None


class SemanticCacheMilvusSearchCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.search."""

    model_config = ConfigDict(extra="forbid")

    params: SemanticCacheMilvusSearchParamsCompatConfig | None = None
    topk: int | None = None
    consistency_level: str | None = None


class SemanticCacheMilvusConnectionPoolCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.performance.connection_pool."""

    model_config = ConfigDict(extra="forbid")

    max_connections: int | None = None
    max_idle_connections: int | None = None
    acquire_timeout: int | None = None


class SemanticCacheMilvusBatchCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.performance.batch."""

    model_config = ConfigDict(extra="forbid")

    insert_batch_size: int | None = None
    timeout: int | None = None


class SemanticCacheMilvusPerformanceCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.performance."""

    model_config = ConfigDict(extra="forbid")

    connection_pool: SemanticCacheMilvusConnectionPoolCompatConfig | None = None
    batch: SemanticCacheMilvusBatchCompatConfig | None = None


class SemanticCacheMilvusTTLCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.data_management.ttl."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    timestamp_field: str | None = None
    cleanup_interval: int | None = None


class SemanticCacheMilvusCompactionCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.data_management.compaction."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    interval: int | None = None


class SemanticCacheMilvusDataManagementCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.data_management."""

    model_config = ConfigDict(extra="forbid")

    ttl: SemanticCacheMilvusTTLCompatConfig | None = None
    compaction: SemanticCacheMilvusCompactionCompatConfig | None = None


class SemanticCacheMilvusLoggingCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.logging."""

    model_config = ConfigDict(extra="forbid")

    level: str | None = None
    enable_query_log: bool | None = None
    enable_metrics: bool | None = None


class SemanticCacheMilvusDevelopmentCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus.development."""

    model_config = ConfigDict(extra="forbid")

    drop_collection_on_startup: bool | None = None
    auto_create_collection: bool | None = None
    verbose_errors: bool | None = None


class SemanticCacheMilvusCompatConfig(BaseModel):
    """Typed schema for semantic_cache.milvus."""

    model_config = ConfigDict(extra="forbid")

    connection: SemanticCacheMilvusConnectionCompatConfig | None = None
    collection: SemanticCacheMilvusCollectionCompatConfig | None = None
    search: SemanticCacheMilvusSearchCompatConfig | None = None
    performance: SemanticCacheMilvusPerformanceCompatConfig | None = None
    data_management: SemanticCacheMilvusDataManagementCompatConfig | None = None
    logging: SemanticCacheMilvusLoggingCompatConfig | None = None
    development: SemanticCacheMilvusDevelopmentCompatConfig | None = None


class SemanticCacheCompatConfig(BaseModel):
    """Explicit semantic_cache schema while the wider CLI contract still migrates."""

    model_config = ConfigDict(extra="forbid")

    backend_type: str | None = None
    enabled: bool | None = None
    similarity_threshold: float | None = None
    max_entries: int | None = None
    ttl_seconds: int | None = None
    eviction_policy: str | None = None
    use_hnsw: bool | None = None
    hnsw_m: int | None = None
    hnsw_ef_construction: int | None = None
    max_memory_entries: int | None = None
    redis: SemanticCacheRedisCompatConfig | None = None
    milvus: SemanticCacheMilvusCompatConfig | None = None
    backend_config_path: str | None = None
    embedding_model: str | None = None
