package vllm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
)

const vllmMetricsFixture = `
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="qwen3",engine="0"} 4
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="qwen3",engine="0"} 2
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="qwen3",engine="0"} 0.42
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{model_name="qwen3",engine="0"} 10
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{model_name="qwen3",engine="0"} 7
# TYPE vllm:external_prefix_cache_queries counter
vllm:external_prefix_cache_queries{model_name="qwen3",engine="0"} 5
# TYPE vllm:external_prefix_cache_hits counter
vllm:external_prefix_cache_hits{model_name="qwen3",engine="0"} 3
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="qwen3",engine="0",le="0.1"} 1
vllm:time_to_first_token_seconds_bucket{model_name="qwen3",engine="0",le="0.5"} 8
vllm:time_to_first_token_seconds_bucket{model_name="qwen3",engine="0",le="1"} 10
vllm:time_to_first_token_seconds_bucket{model_name="qwen3",engine="0",le="+Inf"} 10
vllm:time_to_first_token_seconds_sum{model_name="qwen3",engine="0"} 3
vllm:time_to_first_token_seconds_count{model_name="qwen3",engine="0"} 10
# TYPE vllm:request_time_per_output_token_seconds histogram
vllm:request_time_per_output_token_seconds_bucket{model_name="qwen3",engine="0",le="0.01"} 3
vllm:request_time_per_output_token_seconds_bucket{model_name="qwen3",engine="0",le="0.03"} 10
vllm:request_time_per_output_token_seconds_bucket{model_name="qwen3",engine="0",le="+Inf"} 10
vllm:request_time_per_output_token_seconds_sum{model_name="qwen3",engine="0"} 0.2
vllm:request_time_per_output_token_seconds_count{model_name="qwen3",engine="0"} 10
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{model_name="qwen3",engine="0",le="1"} 4
vllm:e2e_request_latency_seconds_bucket{model_name="qwen3",engine="0",le="3"} 10
vllm:e2e_request_latency_seconds_bucket{model_name="qwen3",engine="0",le="+Inf"} 10
vllm:e2e_request_latency_seconds_sum{model_name="qwen3",engine="0"} 20
vllm:e2e_request_latency_seconds_count{model_name="qwen3",engine="0"} 10
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_bucket{model_name="qwen3",engine="0",le="0.05"} 9
vllm:request_queue_time_seconds_bucket{model_name="qwen3",engine="0",le="0.2"} 10
vllm:request_queue_time_seconds_bucket{model_name="qwen3",engine="0",le="+Inf"} 10
vllm:request_queue_time_seconds_sum{model_name="qwen3",engine="0"} 0.3
vllm:request_queue_time_seconds_count{model_name="qwen3",engine="0"} 10
`

func TestBuildTelemetrySamplesMapsVLLMMetrics(t *testing.T) {
	families, err := parseMetricFamilies(vllmMetricsFixture)
	if err != nil {
		t.Fatalf("parseMetricFamilies() error = %v", err)
	}
	target := backend.AdapterTarget{
		Identity:        backend.BackendIdentity{BackendID: "backend-a", ModelName: "qwen3", Endpoint: "primary"},
		MetricsEndpoint: "http://127.0.0.1:8000/metrics",
	}

	samples := buildTelemetrySamples(target, families, 5*time.Second, time.Unix(100, 0))
	if len(samples) != 1 {
		t.Fatalf("len(samples) = %d, want 1: %#v", len(samples), samples)
	}
	got := samples[0]
	if got.Identity.EngineKind != backend.EngineKindVLLM {
		t.Fatalf("EngineKind = %q, want vllm", got.Identity.EngineKind)
	}
	if got.Identity.ReplicaID != "0" {
		t.Fatalf("ReplicaID = %q, want 0", got.Identity.ReplicaID)
	}
	requireIntPointer(t, "QueueDepth", got.QueueDepth, 4)
	requireIntPointer(t, "ActiveRequests", got.ActiveRequests, 2)
	requireFloatPointer(t, "KVCachePressure", got.KVCachePressure, 0.42)
	requireFloatPointer(t, "PrefixCacheHitRate", got.Affinity.PrefixCacheHitRate, 10.0/15.0)
	requireFloatPointer(t, "TTFT p50", got.Latency.TTFTSeconds.P50Seconds, 0.5)
	requireFloatPointer(t, "queue p90", got.Latency.QueueSeconds.P90Seconds, 0.05)
}

func requireIntPointer(t *testing.T, name string, got *int, want int) {
	t.Helper()
	if got == nil {
		t.Fatalf("%s = nil, want %d", name, want)
	}
	if *got != want {
		t.Fatalf("%s = %d, want %d", name, *got, want)
	}
}

func requireFloatPointer(t *testing.T, name string, got *float64, want float64) {
	t.Helper()
	if got == nil {
		t.Fatalf("%s = nil, want %f", name, want)
	}
	if *got != want {
		t.Fatalf("%s = %f, want %f", name, *got, want)
	}
}

func TestAdapterCollectScrapesHTTPMetrics(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want bearer token", got)
		}
		fmt.Fprint(w, vllmMetricsFixture)
	}))
	defer server.Close()

	adapter, err := NewAdapter(backend.AdapterConfig{
		Targets: []backend.AdapterTarget{{
			Identity:        backend.BackendIdentity{BackendID: "backend-a", ModelName: "qwen3"},
			MetricsEndpoint: server.URL,
			Headers:         map[string]string{"Authorization": "Bearer test-key"},
		}},
		TTL:            7 * time.Second,
		RequestTimeout: time.Second,
	})
	if err != nil {
		t.Fatalf("NewAdapter() error = %v", err)
	}

	samples, err := adapter.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect() error = %v", err)
	}
	if len(samples) != 1 {
		t.Fatalf("len(samples) = %d, want 1", len(samples))
	}
	if samples[0].TTL != 7*time.Second {
		t.Fatalf("TTL = %s, want 7s", samples[0].TTL)
	}
}

func TestAdapterCollectFailsWhenNoRecognizedMetrics(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "# TYPE unrelated gauge\nunrelated 1\n")
	}))
	defer server.Close()

	adapter, err := NewAdapter(backend.AdapterConfig{
		Targets: []backend.AdapterTarget{{
			Identity:        backend.BackendIdentity{BackendID: "backend-a", ModelName: "qwen3"},
			MetricsEndpoint: server.URL,
		}},
	})
	if err != nil {
		t.Fatalf("NewAdapter() error = %v", err)
	}

	_, err = adapter.Collect(context.Background())
	if err == nil || !strings.Contains(err.Error(), "no recognized") {
		t.Fatalf("Collect() error = %v, want no recognized metrics", err)
	}
}
