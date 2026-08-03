package tensorrtllm

import (
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
)

// trtLLMBaselineMetrics is a representative Triton TensorRT-LLM /metrics
// document (batch-manager gauges + base Triton counters), modeled on the
// sample in the Triton tensorrtllm_backend README.
const trtLLMBaselineMetrics = `
# HELP nv_trt_llm_request_metrics TRT LLM request metrics
# TYPE nv_trt_llm_request_metrics gauge
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="waiting",version="1"} 3
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="context",version="1"} 1
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="scheduled",version="1"} 4
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="max",version="1"} 512
nv_trt_llm_request_metrics{model="tensorrt_llm",request_type="active",version="1"} 7
# HELP nv_trt_llm_kv_cache_block_metrics TRT LLM KV cache block metrics
# TYPE nv_trt_llm_kv_cache_block_metrics gauge
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="fraction",model="tensorrt_llm",version="1"} 0.6
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="used",model="tensorrt_llm",version="1"} 2000
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="free",model="tensorrt_llm",version="1"} 4000
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="max",model="tensorrt_llm",version="1"} 8000
# HELP nv_trt_llm_disaggregated_serving_metrics TRT LLM disaggregated serving metrics
# TYPE nv_trt_llm_disaggregated_serving_metrics counter
nv_trt_llm_disaggregated_serving_metrics{disaggregated_serving_type="kv_cache_transfer_ms",model="tensorrt_llm",version="1"} 12
# HELP nv_gpu_utilization GPU utilization
# TYPE nv_gpu_utilization gauge
nv_gpu_utilization{gpu_uuid="GPU-0"} 0.4
nv_gpu_utilization{gpu_uuid="GPU-1"} 0.75
# HELP nv_gpu_memory_used_bytes GPU used memory
# TYPE nv_gpu_memory_used_bytes gauge
nv_gpu_memory_used_bytes{gpu_uuid="GPU-0"} 4000
nv_gpu_memory_used_bytes{gpu_uuid="GPU-1"} 6000
# HELP nv_gpu_memory_total_bytes GPU total memory
# TYPE nv_gpu_memory_total_bytes gauge
nv_gpu_memory_total_bytes{gpu_uuid="GPU-0"} 10000
nv_gpu_memory_total_bytes{gpu_uuid="GPU-1"} 10000
# HELP nv_inference_request_duration_us request duration
# TYPE nv_inference_request_duration_us counter
nv_inference_request_duration_us{model="tensorrt_llm",version="1"} 2000000
# HELP nv_inference_queue_duration_us queue duration
# TYPE nv_inference_queue_duration_us counter
nv_inference_queue_duration_us{model="tensorrt_llm",version="1"} 500000
# HELP nv_inference_request_success request success count
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="tensorrt_llm",version="1"} 100
`

func parseFixture(t *testing.T, text string) map[string]*dto.MetricFamily {
	t.Helper()
	families, err := parseMetricFamilies(text)
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return families
}

func testTarget() backend.AdapterTarget {
	return backend.AdapterTarget{
		Identity: backend.BackendIdentity{
			BackendID: "trtllm-a",
			ModelName: "llama-3.1-70b",
			ReplicaID: "10.0.0.5:8000",
			Endpoint:  "trtllm-a",
		},
		MetricsEndpoint: "http://10.0.0.5:8002/metrics",
	}
}

// wantFloatPtr fails when a *float64 is nil or not equal to want.
func wantFloatPtr(t *testing.T, name string, got *float64, want float64) {
	t.Helper()
	if got == nil {
		t.Errorf("%s = nil, want %v", name, want)
		return
	}
	if *got != want {
		t.Errorf("%s = %v, want %v", name, *got, want)
	}
}

// wantIntPtr fails when an *int is nil or not equal to want.
func wantIntPtr(t *testing.T, name string, got *int, want int) {
	t.Helper()
	if got == nil {
		t.Errorf("%s = nil, want %d", name, want)
		return
	}
	if *got != want {
		t.Errorf("%s = %d, want %d", name, *got, want)
	}
}

func TestNormalizeTargetBaseline(t *testing.T) {
	families := parseFixture(t, trtLLMBaselineMetrics)
	now := time.Unix(1000, 0)
	sample, _, recognized := normalizeTarget(testTarget(), families, counterSnapshot{}, 5*time.Second, now)
	if !recognized {
		t.Fatal("expected recognized metrics")
	}
	if sample.Identity.EngineKind != backend.EngineKindTensorRTLLM {
		t.Errorf("engine kind = %q, want tensorrt-llm", sample.Identity.EngineKind)
	}
	wantIntPtr(t, "QueueDepth", sample.QueueDepth, 3)
	wantIntPtr(t, "ActiveRequests", sample.ActiveRequests, 7)
	wantFloatPtr(t, "KVCachePressure", sample.KVCachePressure, 0.6)
	// reuse = 1 - used/max = 1 - 2000/8000 = 0.75
	wantFloatPtr(t, "KVCacheReuseScore", sample.Affinity.KVCacheReuseScore, 0.75)
	// GPU util = max(0.4, 0.75) = 0.75
	wantFloatPtr(t, "GPUUtilization", sample.GPUUtilization, 0.75)
	// mem = (4000+6000)/(10000+10000) = 0.5
	wantFloatPtr(t, "MemoryPressure", sample.MemoryPressure, 0.5)
	if got := sample.Affinity.ExtraHints["kv_cache_transfer_ms"]; got != 12 {
		t.Errorf("kv_cache_transfer_ms hint = %v, want 12", got)
	}
	if sample.Health != backend.HealthStateHealthy {
		t.Errorf("Health = %q, want healthy (not saturated)", sample.Health)
	}
	if sample.Confidence != 1 {
		t.Errorf("Confidence = %v, want 1 (nv_trt_llm_* present)", sample.Confidence)
	}
	// Baseline: no previous snapshot, so counter-average latency stays nil.
	if sample.Latency.E2ESeconds.P50Seconds != nil {
		t.Errorf("E2E P50 = %v, want nil on first scrape", *sample.Latency.E2ESeconds.P50Seconds)
	}
}

func TestCounterDeltaLatencyAcrossScrapes(t *testing.T) {
	families := parseFixture(t, trtLLMBaselineMetrics)
	// prev snapshot: 1.0s total over 50 reqs earlier.
	prev := counterSnapshot{
		requestDurationUs: 1000000,
		queueDurationUs:   250000,
		requestCount:      50,
		valid:             true,
	}
	sample, _, _ := normalizeTarget(testTarget(), families, prev, 5*time.Second, time.Unix(1000, 0))
	// delta request dur = 2000000-1000000 = 1e6 us over 100-50=50 reqs -> 20000us = 0.02s
	if sample.Latency.E2ESeconds.P50Seconds == nil {
		t.Fatal("expected E2E average from counter delta")
	}
	if got := *sample.Latency.E2ESeconds.P50Seconds; got != 0.02 {
		t.Errorf("E2E avg = %v, want 0.02", got)
	}
	// queue delta = 500000-250000 = 250000 us / 50 = 5000us = 0.005s
	if sample.Latency.QueueSeconds.P50Seconds == nil || *sample.Latency.QueueSeconds.P50Seconds != 0.005 {
		t.Errorf("queue avg = %v, want 0.005", sample.Latency.QueueSeconds.P50Seconds)
	}
}

func TestSaturationDegradesHealth(t *testing.T) {
	metrics := `
# TYPE nv_trt_llm_request_metrics gauge
nv_trt_llm_request_metrics{request_type="waiting"} 5
nv_trt_llm_request_metrics{request_type="active"} 10
# TYPE nv_trt_llm_kv_cache_block_metrics gauge
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="fraction"} 0.98
`
	families := parseFixture(t, metrics)
	sample, _, recognized := normalizeTarget(testTarget(), families, counterSnapshot{}, 5*time.Second, time.Unix(1, 0))
	if !recognized {
		t.Fatal("expected recognized")
	}
	if sample.Health != backend.HealthStateDegraded {
		t.Errorf("Health = %q, want degraded (waiting>0 && kv>=0.95)", sample.Health)
	}
}

func TestConfidenceDowngradeWithoutTRTLLMMetrics(t *testing.T) {
	// Only base Triton core metrics, no nv_trt_llm_* families.
	metrics := `
# TYPE nv_inference_pending_request_count gauge
nv_inference_pending_request_count{model="m"} 2
# TYPE nv_gpu_utilization gauge
nv_gpu_utilization{gpu_uuid="GPU-0"} 0.3
`
	families := parseFixture(t, metrics)
	sample, _, recognized := normalizeTarget(testTarget(), families, counterSnapshot{}, 5*time.Second, time.Unix(1, 0))
	if !recognized {
		t.Fatal("expected recognized from base Triton metrics")
	}
	if sample.QueueDepth == nil || *sample.QueueDepth != 2 {
		t.Errorf("QueueDepth = %v, want 2 (pending fallback)", sample.QueueDepth)
	}
	if sample.Confidence != 0.5 {
		t.Errorf("Confidence = %v, want 0.5 (no nv_trt_llm_*)", sample.Confidence)
	}
}

func TestUnrecognizedMetricsReturnsFalse(t *testing.T) {
	families := parseFixture(t, "# TYPE some_other_metric gauge\nsome_other_metric 1\n")
	_, _, recognized := normalizeTarget(testTarget(), families, counterSnapshot{}, 5*time.Second, time.Unix(1, 0))
	if recognized {
		t.Error("expected recognized=false for non-TRT-LLM/Triton metrics")
	}
}

func TestTTFTHistogramTierWhenPresent(t *testing.T) {
	metrics := `
# TYPE nv_trt_llm_request_metrics gauge
nv_trt_llm_request_metrics{request_type="active"} 1
# TYPE nv_inference_first_response_histogram_ms histogram
nv_inference_first_response_histogram_ms_count{model="m",version="1"} 37
nv_inference_first_response_histogram_ms_sum{model="m",version="1"} 10771
nv_inference_first_response_histogram_ms_bucket{model="m",version="1",le="100"} 8
nv_inference_first_response_histogram_ms_bucket{model="m",version="1",le="500"} 30
nv_inference_first_response_histogram_ms_bucket{model="m",version="1",le="2000"} 36
nv_inference_first_response_histogram_ms_bucket{model="m",version="1",le="5000"} 37
nv_inference_first_response_histogram_ms_bucket{model="m",version="1",le="+Inf"} 37
`
	families := parseFixture(t, metrics)
	sample, _, recognized := normalizeTarget(testTarget(), families, counterSnapshot{}, 5*time.Second, time.Unix(1, 0))
	if !recognized {
		t.Fatal("expected recognized")
	}
	// TTFT percentiles should be filled from the histogram, converted ms->s.
	if sample.Latency.TTFTSeconds.P50Seconds == nil {
		t.Fatal("expected TTFT P50 from histogram")
	}
	// rank for P50 = 0.5*37 = 18.5 -> first bucket with cum>=18.5 is le=500ms -> 0.5s
	if got := *sample.Latency.TTFTSeconds.P50Seconds; got != 0.5 {
		t.Errorf("TTFT P50 = %v s, want 0.5", got)
	}
}
