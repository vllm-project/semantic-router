"""Prometheus metrics of one runtime process (no request content in any label)."""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
)
ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
TOKEN_BUCKETS = (64, 256, 1024, 4096, 16384, 65536, 262144)


class RuntimeMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests = Counter(
            "vllm_srun_requests",
            "API requests by endpoint and HTTP status.",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.questions = Counter(
            "vllm_srun_questions",
            "Questions by type and outcome (answered or an error code).",
            ["type", "outcome"],
            registry=self.registry,
        )
        self.input_tokens = Counter(
            "vllm_srun_input_tokens",
            "Rendered input tokens of answered requests.",
            registry=self.registry,
        )
        self.request_seconds = Histogram(
            "vllm_srun_request_duration_seconds",
            "End-to-end request latency.",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.queue_seconds = Histogram(
            "vllm_srun_queue_duration_seconds",
            "Time a job waited before its first forward.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.forward_seconds = Histogram(
            "vllm_srun_forward_duration_seconds",
            "Model forward plus readout per batch.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.batch_rows = Histogram(
            "vllm_srun_batch_rows",
            "Questions per forward.",
            buckets=ROW_BUCKETS,
            registry=self.registry,
        )
        self.batch_tokens = Histogram(
            "vllm_srun_batch_tokens",
            "Unpadded tokens per forward.",
            buckets=TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.deadline_drops = Counter(
            "vllm_srun_deadline_dropped_questions",
            "Questions not run because their deadline passed in the queue.",
            registry=self.registry,
        )
        self.result_cache = Counter(
            "vllm_srun_result_cache",
            "Cacheable item lookups by model and outcome (hit or miss).",
            ["model", "outcome"],
            registry=self.registry,
        )
        self.bundle_tasks = Histogram(
            "vllm_srun_bundle_tasks",
            "Tasks per /v1/bundle request.",
            buckets=ROW_BUCKETS,
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "vllm_srun_queue_depth",
            "Jobs waiting for the model worker.",
            registry=self.registry,
        )
        self.ready = Gauge(
            "vllm_srun_ready",
            "1 when the model passed its golden check.",
            registry=self.registry,
        )
        self.model_memory = Gauge(
            "vllm_srun_model_memory_bytes",
            "Bytes of a model's loaded weights, packed layouts and reduced-precision copies.",
            ["model"],
            registry=self.registry,
        )
        self.model_info = Gauge(
            "vllm_srun_model_info",
            "The served model (value 1).",
            [
                "model",
                "revision",
                "family",
                "engine",
                "accelerator",
                "device",
                "profile",
            ],
            registry=self.registry,
        )

    def observe(self, event: str, values: dict[str, Any]) -> None:
        if event == "queue":
            self.queue_seconds.observe(values["seconds"])
        elif event == "forward":
            self.forward_seconds.observe(values["seconds"])
            self.batch_rows.observe(values["rows"])
            self.batch_tokens.observe(values["tokens"])
        elif event == "deadline":
            self.deadline_drops.inc(values["questions"])

    def render(self) -> bytes:
        return generate_latest(self.registry)
