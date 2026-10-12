package vllm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
)

// exposition renders the vLLM metrics this adapter reads. hits/queries and
// the TTFT histogram are cumulative, as vLLM reports them.
func exposition(waiting, running int, kv float64, ttftSum float64, ttftCount int, hits, queries int) string {
	return fmt.Sprintf(`# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="qwen3-8b"} %d
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="qwen3-8b"} %d
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="qwen3-8b"} %g
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",model_name="qwen3-8b",le="+Inf"} %d
vllm:time_to_first_token_seconds_sum{engine="0",model_name="qwen3-8b"} %g
vllm:time_to_first_token_seconds_count{engine="0",model_name="qwen3-8b"} %d
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="qwen3-8b"} %d
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="qwen3-8b"} %d
`, waiting, running, kv, ttftCount, ttftSum, ttftCount, hits, queries)
}

type metricsServer struct {
	mu   sync.Mutex
	body string
	code int
	srv  *httptest.Server
}

func newMetricsServer(t *testing.T, body string) *metricsServer {
	m := &metricsServer{body: body, code: http.StatusOK}
	m.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.Lock()
		defer m.mu.Unlock()
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(m.code)
		_, _ = w.Write([]byte(m.body))
	}))
	t.Cleanup(m.srv.Close)
	return m
}

func (m *metricsServer) set(body string, code int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.body, m.code = body, code
}

func target(ref, url string) backendtelemetry.Target {
	return backendtelemetry.Target{
		Identity:   backendtelemetry.Identity{Model: "qwen3-8b", BackendRef: ref, Engine: backendtelemetry.EngineVLLM},
		MetricsURL: url + "/metrics",
	}
}

func newTestAdapter(t *testing.T, targets ...backendtelemetry.Target) (*adapter, *time.Time) {
	t.Helper()
	built, err := New(backendtelemetry.AdapterConfig{Targets: targets, TTL: 6 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	a := built.(*adapter)
	now := time.Unix(1_000, 0)
	a.now = func() time.Time { return now }
	return a, &now
}

func TestCollectMapsGauges(t *testing.T) {
	server := newMetricsServer(t, exposition(7, 3, 0.42, 10, 5, 2, 4))
	a, _ := newTestAdapter(t, target("gpu-a", server.srv.URL))
	observations, err := a.Collect(context.Background())
	if err != nil || len(observations) != 1 {
		t.Fatalf("Collect() = %v, %v", observations, err)
	}
	o := observations[0]
	if err := o.Validate(); err != nil {
		t.Fatalf("observation does not satisfy the contract: %v", err)
	}
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
	if o.TTFT.Present() || o.PrefixCacheHitRate.Present() {
		t.Fatal("a first scrape reported window means it cannot know yet")
	}
}

func TestCollectDerivesWindowMeansFromDeltas(t *testing.T) {
	server := newMetricsServer(t, exposition(0, 0, 0, 10, 5, 2, 4))
	a, now := newTestAdapter(t, target("gpu-a", server.srv.URL))
	if _, err := a.Collect(context.Background()); err != nil {
		t.Fatal(err)
	}
	// Four more requests took 2s to first token in total; 3 of 6 lookups hit.
	server.set(exposition(0, 0, 0, 12, 9, 5, 10), http.StatusOK)
	*now = now.Add(2 * time.Second)
	observations, err := a.Collect(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	o := observations[0]
	if v, ok := o.TTFT.Get(); !ok || v != 500*time.Millisecond {
		t.Fatalf("ttft = %v, %v; want 500ms", v, ok)
	}
	if o.LatencyWindow != 2*time.Second {
		t.Fatalf("latency window = %v", o.LatencyWindow)
	}
	if v, ok := o.PrefixCacheHitRate.Get(); !ok || v != 0.5 {
		t.Fatalf("prefix cache hit rate = %v, %v", v, ok)
	}
	if o.TPOT.Present() || o.E2E.Present() {
		t.Fatal("latencies the engine did not expose were reported")
	}
	if err := o.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestCollectDropsWindowMeansAcrossACounterReset(t *testing.T) {
	server := newMetricsServer(t, exposition(0, 0, 0, 10, 5, 4, 8))
	a, now := newTestAdapter(t, target("gpu-a", server.srv.URL))
	if _, err := a.Collect(context.Background()); err != nil {
		t.Fatal(err)
	}
	server.set(exposition(1, 0, 0, 1, 1, 0, 1), http.StatusOK) // the engine restarted
	*now = now.Add(time.Second)
	observations, err := a.Collect(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	o := observations[0]
	if o.TTFT.Present() || o.PrefixCacheHitRate.Present() || o.LatencyWindow != 0 {
		t.Fatalf("a counter reset produced window means: %+v", o)
	}
	if v, _ := o.QueueDepth.Get(); v != 1 {
		t.Fatalf("queue depth = %d; gauges still apply after a reset", v)
	}
}

func TestCollectReportsFailedTargetsAndKeepsTheRest(t *testing.T) {
	healthy := newMetricsServer(t, exposition(2, 1, 0.1, 0, 0, 0, 0))
	broken := newMetricsServer(t, "")
	broken.set("oops", http.StatusInternalServerError)
	garbled := newMetricsServer(t, "not a metric line {")
	a, _ := newTestAdapter(t,
		target("gpu-a", healthy.srv.URL), target("gpu-b", broken.srv.URL), target("gpu-c", garbled.srv.URL))
	observations, err := a.Collect(context.Background())
	if len(observations) != 1 || observations[0].Identity.BackendRef != "gpu-a" {
		t.Fatalf("observations = %+v", observations)
	}
	if err == nil || !strings.Contains(err.Error(), "gpu-b") || !strings.Contains(err.Error(), "gpu-c") {
		t.Fatalf("err = %v, want both failed targets named", err)
	}
}

func TestCollectWithoutMetricsReportsAbsence(t *testing.T) {
	server := newMetricsServer(t, "# TYPE process_cpu_seconds_total counter\nprocess_cpu_seconds_total 1\n")
	a, _ := newTestAdapter(t, target("gpu-a", server.srv.URL))
	observations, err := a.Collect(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	o := observations[0]
	if o.QueueDepth.Present() || o.ActiveRequests.Present() || o.KVCacheUsage.Present() {
		t.Fatalf("metrics the engine never exposed were reported: %+v", o)
	}
}

func TestDataParallelGaugesSumCountsAndAverageUsage(t *testing.T) {
	body := `# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0"} 2
vllm:num_requests_waiting{engine="1"} 3
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{engine="0"} 0.2
vllm:gpu_cache_usage_perc{engine="1"} 0.6
`
	server := newMetricsServer(t, body)
	a, _ := newTestAdapter(t, target("gpu-a", server.srv.URL))
	observations, err := a.Collect(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	o := observations[0]
	if v, _ := o.QueueDepth.Get(); v != 5 {
		t.Fatalf("queue depth = %d, want the engines' sum 5", v)
	}
	if v, _ := o.KVCacheUsage.Get(); v < 0.3999 || v > 0.4001 {
		t.Fatalf("kv cache usage = %v, want the engines' mean 0.4 from the legacy metric", v)
	}
}

func TestNewRejectsForeignTargets(t *testing.T) {
	foreign := backendtelemetry.Target{
		Identity:   backendtelemetry.Identity{Model: "m", BackendRef: "b", Engine: backendtelemetry.EngineSGLang},
		MetricsURL: "http://127.0.0.1:1/metrics",
	}
	if _, err := New(backendtelemetry.AdapterConfig{Targets: []backendtelemetry.Target{foreign}}); err == nil {
		t.Fatal("accepted an SGLang target")
	}
}
