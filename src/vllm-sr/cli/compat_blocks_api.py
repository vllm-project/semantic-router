"""Typed API compatibility blocks for the TD001 CLI migration."""

from pydantic import BaseModel, ConfigDict


class BatchSizeRangeCompatConfig(BaseModel):
    """Typed schema for api.batch_classification.metrics.batch_size_ranges."""

    model_config = ConfigDict(extra="forbid")

    min: int | None = None
    max: int | None = None
    label: str | None = None


class BatchClassificationMetricsCompatConfig(BaseModel):
    """Typed schema for api.batch_classification.metrics."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    detailed_goroutine_tracking: bool | None = None
    high_resolution_timing: bool | None = None
    sample_rate: float | None = None
    batch_size_ranges: list[BatchSizeRangeCompatConfig] | None = None
    duration_buckets: list[float] | None = None
    size_buckets: list[int] | None = None


class APIBatchClassificationCompatConfig(BaseModel):
    """Typed schema for api.batch_classification."""

    model_config = ConfigDict(extra="forbid")

    max_batch_size: int | None = None
    concurrency_threshold: int | None = None
    max_concurrency: int | None = None
    metrics: BatchClassificationMetricsCompatConfig | None = None


class APICompatConfig(BaseModel):
    """Typed schema for api."""

    model_config = ConfigDict(extra="forbid")

    batch_classification: APIBatchClassificationCompatConfig | None = None
