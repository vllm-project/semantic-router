package tensorrtllm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
)

func TestNewAdapterValidation(t *testing.T) {
	if _, err := NewAdapter(backend.AdapterConfig{}); err == nil {
		t.Error("expected error for no targets")
	}
	// Missing backend_id.
	_, err := NewAdapter(backend.AdapterConfig{Targets: []backend.AdapterTarget{{
		Identity:        backend.BackendIdentity{ModelName: "m"},
		MetricsEndpoint: "http://x/metrics",
	}}})
	if err == nil {
		t.Error("expected error for missing backend_id")
	}
}

func TestEngineKind(t *testing.T) {
	a, err := NewAdapter(backend.AdapterConfig{Targets: []backend.AdapterTarget{testTarget()}})
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	if a.EngineKind() != backend.EngineKindTensorRTLLM {
		t.Errorf("EngineKind = %q, want tensorrt-llm", a.EngineKind())
	}
}

func TestCollectEndToEnd(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte(trtLLMBaselineMetrics))
	}))
	defer srv.Close()

	target := testTarget()
	target.MetricsEndpoint = srv.URL + "/metrics"

	a, err := NewAdapter(backend.AdapterConfig{
		Targets: []backend.AdapterTarget{target},
		TTL:     5 * time.Second,
	})
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}

	samples, err := a.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(samples) != 1 {
		t.Fatalf("expected 1 sample, got %d", len(samples))
	}
	s := samples[0]
	if s.QueueDepth == nil || *s.QueueDepth != 3 {
		t.Errorf("QueueDepth = %v, want 3", s.QueueDepth)
	}
	if s.Identity.EngineKind != backend.EngineKindTensorRTLLM {
		t.Errorf("EngineKind = %q", s.Identity.EngineKind)
	}
}

func TestCollectScrapeErrorFailsOpen(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	target := testTarget()
	target.MetricsEndpoint = srv.URL + "/metrics"
	a, _ := NewAdapter(backend.AdapterConfig{Targets: []backend.AdapterTarget{target}})

	samples, err := a.Collect(context.Background())
	if err == nil {
		t.Error("expected scrape error")
	}
	if len(samples) != 0 {
		t.Errorf("expected 0 samples on error, got %d", len(samples))
	}
}

func TestCollectAndStoreFreshnessTTL(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(trtLLMBaselineMetrics))
	}))
	defer srv.Close()

	target := testTarget()
	target.MetricsEndpoint = srv.URL + "/metrics"
	a, _ := NewAdapter(backend.AdapterConfig{
		Targets: []backend.AdapterTarget{target},
		TTL:     50 * time.Millisecond,
	})

	store := backend.NewStore(50 * time.Millisecond)
	samples, err := a.Collect(context.Background())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if err := store.UpsertMany(samples); err != nil {
		t.Fatalf("UpsertMany: %v", err)
	}

	fresh := store.ListFreshByModel("llama-3.1-70b")
	if len(fresh) != 1 {
		t.Fatalf("expected 1 fresh sample, got %d", len(fresh))
	}

	// After TTL elapses the sample is stale -> fail open (empty).
	time.Sleep(80 * time.Millisecond)
	if stale := store.ListFreshByModel("llama-3.1-70b"); len(stale) != 0 {
		t.Errorf("expected 0 fresh samples after TTL, got %d", len(stale))
	}
}

func TestRegister(t *testing.T) {
	if err := Register(); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if !backend.AdapterRegistered(backend.EngineKindTensorRTLLM) {
		t.Error("expected tensorrt-llm adapter to be registered")
	}
	a, err := backend.NewAdapter(backend.EngineKindTensorRTLLM, backend.AdapterConfig{
		Targets: []backend.AdapterTarget{testTarget()},
	})
	if err != nil {
		t.Fatalf("NewAdapter via registry: %v", err)
	}
	if a.EngineKind() != backend.EngineKindTensorRTLLM {
		t.Errorf("EngineKind = %q", a.EngineKind())
	}
}
