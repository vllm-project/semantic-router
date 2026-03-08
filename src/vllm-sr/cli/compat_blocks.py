"""Typed top-level compatibility blocks for the TD001 CLI migration."""

from typing import Any

from pydantic import BaseModel, ConfigDict

_TYPED_COMPAT_BLOCKS_ATTR = "_td001_typed_compat_blocks"


class PromptGuardCompatConfig(BaseModel):
    """Explicit prompt_guard schema while the wider CLI contract still migrates."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    model_id: str | None = None
    threshold: float | None = None
    use_cpu: bool | None = None
    use_modernbert: bool | None = None
    use_mmbert_32k: bool | None = None
    jailbreak_mapping_path: str | None = None
    use_vllm: bool | None = None


class ToolFilteringWeightsCompatConfig(BaseModel):
    """Typed schema for tools.advanced_filtering.weights."""

    model_config = ConfigDict(extra="forbid")

    embed: float | None = None
    lexical: float | None = None
    tag: float | None = None
    name: float | None = None
    category: float | None = None


class AdvancedToolFilteringCompatConfig(BaseModel):
    """Typed schema for tools.advanced_filtering."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    candidate_pool_size: int | None = None
    min_lexical_overlap: int | None = None
    min_combined_score: float | None = None
    weights: ToolFilteringWeightsCompatConfig | None = None
    use_category_filter: bool | None = None
    category_confidence_threshold: float | None = None
    allow_tools: list[str] | None = None
    block_tools: list[str] | None = None


class ToolsCompatConfig(BaseModel):
    """Explicit tools schema while the wider CLI contract still migrates."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    top_k: int | None = None
    similarity_threshold: float | None = None
    tools_db_path: str | None = None
    fallback_to_empty: bool | None = None
    advanced_filtering: AdvancedToolFilteringCompatConfig | None = None


class WindowedMetricsCompatConfig(BaseModel):
    """Typed schema for observability.metrics.windowed_metrics."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    time_windows: list[str] | None = None
    update_interval: str | None = None
    model_metrics: bool | None = None
    queue_depth_estimation: bool | None = None
    max_models: int | None = None


class MetricsCompatConfig(BaseModel):
    """Typed schema for observability.metrics."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    windowed_metrics: WindowedMetricsCompatConfig | None = None


class TracingExporterCompatConfig(BaseModel):
    """Typed schema for observability.tracing.exporter."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = None
    endpoint: str | None = None
    insecure: bool | None = None


class TracingSamplingCompatConfig(BaseModel):
    """Typed schema for observability.tracing.sampling."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = None
    rate: float | None = None


class TracingResourceCompatConfig(BaseModel):
    """Typed schema for observability.tracing.resource."""

    model_config = ConfigDict(extra="forbid")

    service_name: str | None = None
    service_version: str | None = None
    deployment_environment: str | None = None


class TracingCompatConfig(BaseModel):
    """Typed schema for observability.tracing."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    provider: str | None = None
    exporter: TracingExporterCompatConfig | None = None
    sampling: TracingSamplingCompatConfig | None = None
    resource: TracingResourceCompatConfig | None = None


class ObservabilityCompatConfig(BaseModel):
    """Explicit observability schema while the wider CLI contract still migrates."""

    model_config = ConfigDict(extra="forbid")

    metrics: MetricsCompatConfig | None = None
    tracing: TracingCompatConfig | None = None


class LooperCompatConfig(BaseModel):
    """Explicit looper schema while the wider CLI contract still migrates."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    endpoint: str | None = None
    model_endpoints: dict[str, str] | None = None
    grpc_max_msg_size_mb: int | None = None
    timeout_seconds: int | None = None
    retry_count: int | None = None
    headers: dict[str, str] | None = None


class RouterReplayRedisCompatConfig(BaseModel):
    """Typed schema for router_replay.redis."""

    model_config = ConfigDict(extra="forbid")

    address: str | None = None
    db: int | None = None
    password: str | None = None
    use_tls: bool | None = None
    tls_skip_verify: bool | None = None
    max_retries: int | None = None
    pool_size: int | None = None
    key_prefix: str | None = None


class RouterReplayPostgresCompatConfig(BaseModel):
    """Typed schema for router_replay.postgres."""

    model_config = ConfigDict(extra="forbid")

    host: str | None = None
    port: int | None = None
    database: str | None = None
    user: str | None = None
    password: str | None = None
    ssl_mode: str | None = None
    max_open_conns: int | None = None
    max_idle_conns: int | None = None
    conn_max_lifetime: int | None = None
    table_name: str | None = None


class RouterReplayMilvusCompatConfig(BaseModel):
    """Typed schema for router_replay.milvus."""

    model_config = ConfigDict(extra="forbid")

    address: str | None = None
    username: str | None = None
    password: str | None = None
    collection_name: str | None = None
    consistency_level: str | None = None


class RouterReplayCompatConfig(BaseModel):
    """Explicit router_replay schema while the wider CLI contract still migrates."""

    model_config = ConfigDict(extra="forbid")

    store_backend: str | None = None
    ttl_seconds: int | None = None
    async_writes: bool | None = None
    redis: RouterReplayRedisCompatConfig | None = None
    postgres: RouterReplayPostgresCompatConfig | None = None
    milvus: RouterReplayMilvusCompatConfig | None = None


class ResponseAPIMilvusCompatConfig(BaseModel):
    """Typed schema for response_api.milvus."""

    model_config = ConfigDict(extra="forbid")

    address: str | None = None
    database: str | None = None
    collection: str | None = None


class ResponseAPIRedisCompatConfig(BaseModel):
    """Typed schema for response_api.redis."""

    model_config = ConfigDict(extra="forbid")

    address: str | None = None
    password: str | None = None
    db: int | None = None
    key_prefix: str | None = None
    cluster_mode: bool | None = None
    cluster_addresses: list[str] | None = None
    pool_size: int | None = None
    min_idle_conns: int | None = None
    max_retries: int | None = None
    dial_timeout: int | None = None
    read_timeout: int | None = None
    write_timeout: int | None = None
    tls_enabled: bool | None = None
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    tls_ca_path: str | None = None
    config_path: str | None = None


class ResponseAPICompatConfig(BaseModel):
    """Explicit response_api schema while the wider CLI contract still migrates."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    store_backend: str | None = None
    ttl_seconds: int | None = None
    max_responses: int | None = None
    backend_config_path: str | None = None
    milvus: ResponseAPIMilvusCompatConfig | None = None
    redis: ResponseAPIRedisCompatConfig | None = None


class TypedCompatBlocks(BaseModel):
    """Explicitly parsed compatibility blocks that no longer rely on model_extra."""

    model_config = ConfigDict(extra="forbid")

    looper: LooperCompatConfig | None = None
    observability: ObservabilityCompatConfig | None = None
    prompt_guard: PromptGuardCompatConfig | None = None
    response_api: ResponseAPICompatConfig | None = None
    router_replay: RouterReplayCompatConfig | None = None
    tools: ToolsCompatConfig | None = None


def extract_typed_compat_blocks(
    data: dict[str, Any],
) -> tuple[dict[str, Any], TypedCompatBlocks]:
    """Split typed compat blocks from raw CLI config data before UserConfig parsing."""

    typed_input = {}
    sanitized_data = dict(data)

    if "looper" in sanitized_data:
        typed_input["looper"] = sanitized_data.pop("looper")
    if "observability" in sanitized_data:
        typed_input["observability"] = sanitized_data.pop("observability")
    if "prompt_guard" in sanitized_data:
        typed_input["prompt_guard"] = sanitized_data.pop("prompt_guard")
    if "response_api" in sanitized_data:
        typed_input["response_api"] = sanitized_data.pop("response_api")
    if "router_replay" in sanitized_data:
        typed_input["router_replay"] = sanitized_data.pop("router_replay")
    if "tools" in sanitized_data:
        typed_input["tools"] = sanitized_data.pop("tools")

    compat_blocks = TypedCompatBlocks(**typed_input)
    return sanitized_data, compat_blocks


def attach_typed_compat_blocks(
    target: object, compat_blocks: TypedCompatBlocks
) -> None:
    """Attach parsed typed compat blocks to a validated UserConfig instance."""

    setattr(target, _TYPED_COMPAT_BLOCKS_ATTR, compat_blocks)


def get_typed_compat_blocks(target: object) -> TypedCompatBlocks:
    """Read typed compat blocks from a parsed UserConfig instance."""

    compat_blocks = getattr(target, _TYPED_COMPAT_BLOCKS_ATTR, None)
    if compat_blocks is None:
        return TypedCompatBlocks()
    return compat_blocks
