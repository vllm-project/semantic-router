"""Per-transfer measurements using vLLM's connector metrics interface."""

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
)


class MapperStats(KVConnectorStats):
    def reset(self):
        self.data.clear()

    def aggregate(self, other):
        self.data.setdefault("loads", []).extend(other.data.get("loads", []))
        return self

    def reduce(self):
        loads = self.data.get("loads", [])
        return {
            "mapper_loads": len(loads),
            "mapper_tokens_saved": sum(load["tokens"] for load in loads),
            "mapper_bytes_fetched": sum(load["bytes"] for load in loads),
        }

    def is_empty(self):
        return not self.data.get("loads")


class MapperPromMetrics(KVConnectorPromMetrics):
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self.loads = self._counter_cls(
            "vllm:kv_mapper_loads_total",
            "Worker mapper load results",
            labelnames=[*labelnames, "status"],
        )
        self.tokens = self._counter_cls(
            "vllm:kv_mapper_tokens_saved_total",
            "Successfully loaded mapped tokens",
            labelnames=labelnames,
        )
        self.bytes = self._counter_cls(
            "vllm:kv_mapper_bytes_fetched_total",
            "Source KV bytes fetched by workers",
            labelnames=labelnames,
        )
        self.fetch = self._histogram_cls(
            "vllm:kv_mapper_fetch_seconds",
            "Worker source snapshot fetch time",
            labelnames=labelnames,
        )
        self.mapping = self._histogram_cls(
            "vllm:kv_mapper_map_load_seconds",
            "Completed mapping and cache injection time",
            labelnames=labelnames,
        )

    def observe(self, transfer_stats_data, engine_idx=0):
        labels = self.per_engine_labelvalues[engine_idx]
        for load in transfer_stats_data.get("loads", []):
            self.loads.labels(*labels, load["status"]).inc()
            self.tokens.labels(*labels).inc(load["tokens"])
            self.bytes.labels(*labels).inc(load["bytes"])
            self.fetch.labels(*labels).observe(load["fetch_seconds"])
            if load["status"] == "applied":
                self.mapping.labels(*labels).observe(load["map_seconds"])
