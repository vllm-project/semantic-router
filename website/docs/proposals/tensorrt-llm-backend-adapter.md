# TensorRT-LLM / Triton Backend Telemetry Adapter

Status: implementation
Issue: [#2377](https://github.com/vllm-project/semantic-router/issues/2377)
Parent: [#2332](https://github.com/vllm-project/semantic-router/issues/2332) inference-aware backend routing ·
Contract: [#2349](https://github.com/vllm-project/semantic-router/issues/2349) / [PR #2391](https://github.com/vllm-project/semantic-router/pull/2391)

## Overview

This adapter lets vLLM Semantic Router consume **TensorRT-LLM** and **Triton TensorRT-LLM** replica
telemetry through the same engine-neutral backend contract used by vLLM, SGLang, and ATOM. It scrapes
Triton/TensorRT-LLM Prometheus metrics on an interval, normalizes them into `backend.BackendTelemetry`,
and publishes them into the shared TTL store. Logical model selection is unchanged; backend-aware
second-stage selection (a separate issue) can then read fresh, healthy TensorRT-LLM telemetry.

The adapter lives in `src/semantic-router/pkg/backend/tensorrtllm` and holds no routing logic — all
TensorRT-LLM specifics stay in the collection/mapping layer.

## How this differs from NVIDIA Dynamo

This is a **direct engine adapter**. It reads metrics from a TensorRT-LLM engine or a Triton server
hosting the TensorRT-LLM backend, and maps them into normalized telemetry.

NVIDIA Dynamo is a **serving/orchestration layer above engines** (it can front TensorRT-LLM, vLLM, or
SGLang and do its own KV-aware routing). Dynamo integration is tracked separately in
[NVIDIA Dynamo Integration](https://vllm-semantic-router.com/docs/proposals/nvidia-dynamo-integration/) and is intentionally out of scope here. If
you run TensorRT-LLM under Dynamo, use the Dynamo integration path; this adapter is for direct
TensorRT-LLM / Triton TensorRT-LLM deployments.

## Configuration

Backend telemetry is opt-in and fail-open. Enable the collector under
`global.services.observability.backend_telemetry`, and mark each TensorRT-LLM backend with
`engine_kind: tensorrt-llm`. Each backend self-describes its metrics surface (`metrics_port`,
`metrics_path`) — Triton exposes metrics on a **different port** (default `8002`) than the inference
port (default `8000`).

```yaml
providers:
  models:
    - name: llama-3.1-70b
      backend_refs:
        - name: trtllm-a
          backend_id: trtllm-a
          engine_kind: tensorrt-llm
          endpoint: 10.0.0.5:8000     # Triton inference (HTTP) port
          protocol: http
          metrics_port: 8002          # Triton Prometheus metrics port (default 8002)
          metrics_path: /metrics      # default /metrics

global:
  services:
    observability:
      backend_telemetry:
        enabled: true
        poll_interval: 2s
        ttl: 5s
        request_timeout: 2s
```

When `backend_telemetry.enabled` is `false` (the default) or no TensorRT-LLM backends are configured,
router behavior is completely unchanged.

## Telemetry mapping

TensorRT-LLM/Triton packs several signals into a few gauge families disambiguated by a label
dimension (unlike vLLM's one-metric-per-signal shape). The adapter maps:

| Normalized field | Source |
| --- | --- |
| `QueueDepth` | `nv_trt_llm_request_metrics{request_type="waiting"}` (fallback `nv_inference_pending_request_count`) |
| `ActiveRequests` | `nv_trt_llm_request_metrics{request_type="active"}` |
| `KVCachePressure` | `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="fraction"}` |
| `Affinity.KVCacheReuseScore` | `1 - used/max` from `nv_trt_llm_kv_cache_block_metrics` |
| `GPUUtilization` | `nv_gpu_utilization` (max across GPUs) |
| `MemoryPressure` | `nv_gpu_memory_used_bytes / nv_gpu_memory_total_bytes` |
| `Affinity.ExtraHints["kv_cache_transfer_ms"]` | `nv_trt_llm_disaggregated_serving_metrics` (P/D disagg) |
| `Latency.*` | tiered — see below |
| `Health` | `degraded` when saturated (`waiting>0` and KV fraction >= 0.95), else `healthy` |
| `Confidence` | `1.0` with `nv_trt_llm_*` present; `0.5` when only base Triton metrics are available |

### Latency tiers

Triton latency comes at three fidelity levels; the adapter uses the richest available and always
fails open to the baseline:

1. **Counter average** (always on): `nv_inference_request_duration_us` / `nv_inference_queue_duration_us`
   over request counts, computed as a delta across two scrapes. Fills the average as a coarse central
   estimate; no fabricated percentiles.
2. **TTFT histogram** (experimental, `--metrics-config histogram_latencies=true`):
   `nv_inference_first_response_histogram_ms` yields real TTFT percentiles.
3. **Latency summaries** (experimental, `--metrics-config summary_latencies=true`):
   `nv_inference_request_summary_us` / `nv_inference_queue_summary_us` yield E2E/queue quantiles.

Tiers 2 and 3 are flagged experimental by NVIDIA and are treated as opportunistic enrichment only —
the adapter never depends on them.

## Freshness and fail-open

Samples carry a TTL (default 5s). Second-stage policy reads only fresh telemetry
(`Store.GetFresh` / `ListFreshByModel`), so missing or stale TensorRT-LLM telemetry transparently
falls back to existing model-level routing. Scrape failures are logged, never fatal, and never block
the request hot path (collection is fully asynchronous).

## Replica identity

Triton is one process per metrics endpoint, so each configured target maps to exactly one replica.
MIG slices and Triton multi-instance groups are not modeled as sub-replicas.
