package sglang

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
)

// The expositions below follow the label layout of SGLang's scheduler metrics
// (python/sglang/srt/observability/metrics_collector.py): model_name,
// engine_type, tp_rank, pp_rank and moe_ep_rank on every series, dp_rank with
// data parallelism, and priority with priority scheduling.

const singleRank = `# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 3.0
# HELP sglang:num_queue_reqs The number of requests in the waiting queue.
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 7.0
# HELP sglang:cache_hit_rate The prefix cache hit rate.
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 0.9
# HELP sglang:token_usage The token usage.
# TYPE sglang:token_usage gauge
sglang:token_usage{engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 0.42
`

func newMetricsServer(t *testing.T, body string, code int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(code)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func target(ref, url string) backendtelemetry.Target {
	return backendtelemetry.Target{
		Identity:   backendtelemetry.Identity{Model: "qwen3-8b", BackendRef: ref, Engine: backendtelemetry.EngineSGLang},
		MetricsURL: url + "/metrics",
	}
}

func collectOne(t *testing.T, body string) backendtelemetry.Observation {
	t.Helper()
	srv := newMetricsServer(t, body, http.StatusOK)
	built, err := New(backendtelemetry.AdapterConfig{Targets: []backendtelemetry.Target{target("gpu-a", srv.URL)}, TTL: 6 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	observations, err := built.Collect(context.Background())
	if err != nil || len(observations) != 1 {
		t.Fatalf("Collect() = %v, %v", observations, err)
	}
	if err := observations[0].Validate(); err != nil {
		t.Fatalf("observation does not satisfy the contract: %v", err)
	}
	return observations[0]
}

func TestCollectMapsGauges(t *testing.T) {
	o := collectOne(t, singleRank)
	if v, ok := o.QueueDepth.Get(); !ok || v != 7 {
		t.Fatalf("queue depth = %d, %v", v, ok)
	}
	if v, ok := o.ActiveRequests.Get(); !ok || v != 3 {
		t.Fatalf("active requests = %d, %v", v, ok)
	}
	if v, ok := o.KVCacheUsage.Get(); !ok || v != 0.42 {
		t.Fatalf("kv cache usage = %v, %v", v, ok)
	}
	if o.Source != Source || o.TTL != 6*time.Second || o.Health != backendtelemetry.HealthUnknown {
		t.Fatalf("metadata = %q %v %v", o.Source, o.TTL, o.Health)
	}
	if o.PrefixCacheHitRate.Present() || o.TTFT.Present() || o.TPOT.Present() || o.E2E.Present() {
		t.Fatalf("reported a hit rate or latency the adapter does not derive: %+v", o)
	}
}

func TestDataParallelRanksSumCountsAndAverageUsage(t *testing.T) {
	body := `# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{dp_rank="0",engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 2
sglang:num_queue_reqs{dp_rank="1",engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 3
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{dp_rank="0",engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 4
sglang:num_running_reqs{dp_rank="1",engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 6
# TYPE sglang:token_usage gauge
sglang:token_usage{dp_rank="0",engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 0.2
sglang:token_usage{dp_rank="1",engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",tp_rank="0"} 0.6
`
	o := collectOne(t, body)
	if v, _ := o.QueueDepth.Get(); v != 5 {
		t.Fatalf("queue depth = %d, want the ranks' sum 5", v)
	}
	if v, _ := o.ActiveRequests.Get(); v != 10 {
		t.Fatalf("active requests = %d, want the ranks' sum 10", v)
	}
	if v, _ := o.KVCacheUsage.Get(); v < 0.3999 || v > 0.4001 {
		t.Fatalf("kv cache usage = %v, want the ranks' mean 0.4", v)
	}
}

// With --enable-metrics-for-all-schedulers every tensor parallel rank reports
// the same values, and with pipeline parallelism every stage reports. Each
// data-parallel rank still counts once.
func TestEveryReportingSchedulerCountsOncePerRank(t *testing.T) {
	var b strings.Builder
	b.WriteString("# TYPE sglang:num_queue_reqs gauge\n")
	for _, dp := range []string{"0", "1"} {
		for _, pp := range []string{"0", "1"} {
			for _, tp := range []string{"0", "1", "2", "3"} {
				value := "4"
				if pp != "0" || tp != "0" {
					value = "9" // another scheduler's copy; must not be added
				}
				b.WriteString(`sglang:num_queue_reqs{dp_rank="` + dp + `",engine_type="unified",model_name="qwen3-8b",moe_ep_rank="` + tp +
					`",pp_rank="` + pp + `",tp_rank="` + tp + `"} ` + value + "\n")
			}
		}
	}
	o := collectOne(t, b.String())
	if v, _ := o.QueueDepth.Get(); v != 8 {
		t.Fatalf("queue depth = %d, want 4 per data-parallel rank, 8", v)
	}
}

func TestPriorityBreakdownIsNotAdded(t *testing.T) {
	body := `# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",priority="0",tp_rank="0"} 2
sglang:num_queue_reqs{engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",priority="10",tp_rank="0"} 3
sglang:num_queue_reqs{engine_type="unified",model_name="qwen3-8b",moe_ep_rank="0",pp_rank="0",priority="",tp_rank="0"} 5
`
	o := collectOne(t, body)
	if v, _ := o.QueueDepth.Get(); v != 5 {
		t.Fatalf("queue depth = %d, want the priority=\"\" total 5", v)
	}
}

func TestCollectWithoutMetricsReportsAbsence(t *testing.T) {
	o := collectOne(t, "# TYPE process_cpu_seconds_total counter\nprocess_cpu_seconds_total 1\n")
	if o.QueueDepth.Present() || o.ActiveRequests.Present() || o.KVCacheUsage.Present() {
		t.Fatalf("metrics the engine never exposed were reported: %+v", o)
	}
}

func TestInvalidGaugesStayAbsent(t *testing.T) {
	for _, value := range []string{"-1", "NaN", "+Inf", "9.223372036854776e18"} {
		t.Run(value, func(t *testing.T) {
			o := collectOne(t, "# TYPE sglang:num_queue_reqs gauge\nsglang:num_queue_reqs "+value+"\n"+
				"# TYPE sglang:num_running_reqs gauge\nsglang:num_running_reqs "+value+"\n"+
				"# TYPE sglang:token_usage gauge\nsglang:token_usage "+value+"\n")
			if o.QueueDepth.Present() || o.ActiveRequests.Present() || o.KVCacheUsage.Present() {
				t.Fatalf("invalid gauges became observed load: %+v", o)
			}
		})
	}
}

func TestIdleReplicaAndFullCacheAreObserved(t *testing.T) {
	o := collectOne(t, "# TYPE sglang:num_queue_reqs gauge\nsglang:num_queue_reqs 0\n"+
		"# TYPE sglang:token_usage gauge\nsglang:token_usage 1\n")
	if v, ok := o.QueueDepth.Get(); !ok || v != 0 {
		t.Fatalf("queue depth = %d, %v; want an observed 0", v, ok)
	}
	if v, ok := o.KVCacheUsage.Get(); !ok || v != 1 {
		t.Fatalf("kv cache usage = %v, %v; want an observed 1", v, ok)
	}
}

func TestCollectReportsFailedTargetsAndKeepsTheRest(t *testing.T) {
	healthy := newMetricsServer(t, singleRank, http.StatusOK)
	broken := newMetricsServer(t, "oops", http.StatusInternalServerError)
	garbled := newMetricsServer(t, "not a metric line {", http.StatusOK)
	built, err := New(backendtelemetry.AdapterConfig{Targets: []backendtelemetry.Target{
		target("gpu-a", healthy.URL), target("gpu-b", broken.URL), target("gpu-c", garbled.URL),
	}})
	if err != nil {
		t.Fatal(err)
	}
	observations, err := built.Collect(context.Background())
	if len(observations) != 1 || observations[0].Identity.BackendRef != "gpu-a" {
		t.Fatalf("observations = %+v", observations)
	}
	if err == nil || !strings.Contains(err.Error(), "gpu-b") || !strings.Contains(err.Error(), "gpu-c") {
		t.Fatalf("err = %v, want both failed targets named", err)
	}
}

func TestNewRejectsForeignTargets(t *testing.T) {
	foreign := backendtelemetry.Target{
		Identity:   backendtelemetry.Identity{Model: "m", BackendRef: "b", Engine: backendtelemetry.EngineVLLM},
		MetricsURL: "http://127.0.0.1:1/metrics",
	}
	if _, err := New(backendtelemetry.AdapterConfig{Targets: []backendtelemetry.Target{foreign}}); err == nil {
		t.Fatal("accepted a vLLM target")
	}
}

func TestRegisterAddsTheEngine(t *testing.T) {
	registry := backendtelemetry.NewRegistry()
	if err := Register(registry); err != nil {
		t.Fatal(err)
	}
	if !registry.Has(backendtelemetry.EngineSGLang) {
		t.Fatal("sglang is not registered")
	}
}
