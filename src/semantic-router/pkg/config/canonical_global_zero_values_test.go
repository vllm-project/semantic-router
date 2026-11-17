package config

import (
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

func reconcileRoundTrip(t *testing.T, cfg *RouterConfig) *RouterConfig {
	t.Helper()
	canonical := CanonicalStaticConfigFromRouterConfig(cfg)
	encoded, err := yamlv3.Marshal(canonical)
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}
	parsed, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatalf("parse canonical config: %v", err)
	}
	return parsed
}

func TestReconcilePreservesResponseCacheTTLZero(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.SemanticCache.TTLSeconds = 0

	parsed := reconcileRoundTrip(t, cfg)
	if parsed.SemanticCache.TTLSeconds != 0 {
		t.Fatalf("response_cache.ttl_seconds = %d, want 0", parsed.SemanticCache.TTLSeconds)
	}
}

func TestReconcilePreservesRouterReplayTTLZero(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.RouterReplay.TTLSeconds = 0

	parsed := reconcileRoundTrip(t, cfg)
	if parsed.RouterReplay.TTLSeconds != 0 {
		t.Fatalf("router_replay.ttl_seconds = %d, want 0", parsed.RouterReplay.TTLSeconds)
	}
}

func TestReconcilePreservesTracingInsecureFalse(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.Observability.Tracing.Exporter.Insecure = false

	parsed := reconcileRoundTrip(t, cfg)
	if parsed.Observability.Tracing.Exporter.Insecure {
		t.Fatal("tracing.exporter.insecure = true, want false")
	}
}

func TestReconcilePreservesResponseAPIZero(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ResponseAPI.TTLSeconds = 0
	cfg.ResponseAPI.MaxResponses = 0

	parsed := reconcileRoundTrip(t, cfg)
	if parsed.ResponseAPI.TTLSeconds != 0 || parsed.ResponseAPI.MaxResponses != 0 {
		t.Fatalf("response_api = %d/%d, want 0/0",
			parsed.ResponseAPI.TTLSeconds, parsed.ResponseAPI.MaxResponses)
	}
}

func TestReconcilePreservesMemoryRetrievalLimitZero(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.Memory.DefaultRetrievalLimit = 0

	parsed := reconcileRoundTrip(t, cfg)
	if parsed.Memory.DefaultRetrievalLimit != 0 {
		t.Fatalf("memory.default_retrieval_limit = %d, want 0", parsed.Memory.DefaultRetrievalLimit)
	}
}

func TestReconcilePreservesFlowStateTTLZero(t *testing.T) {
	const doc = `
version: v0.5
providers:
  models:
    - name: m
      provider_model_id: m
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
          protocol: http
          weight: 100
recipes:
  - name: default
    strategy: priority
    decisions:
      - name: r
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: m
global:
  integrations:
    looper:
      flow:
        state:
          ttl_seconds: 0
`
	cfg, err := ParseYAMLBytes([]byte(doc))
	if err != nil {
		t.Fatalf("parse source config: %v", err)
	}

	parsed := reconcileRoundTrip(t, cfg)
	if parsed.Looper.Flow.State.TTLSeconds != 0 {
		t.Fatalf("looper.flow.state.ttl_seconds = %d, want 0", parsed.Looper.Flow.State.TTLSeconds)
	}
}

func TestReconcilePreservesTracingSamplingRateZero(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.Observability.Tracing.Sampling.Rate = 0

	parsed := reconcileRoundTrip(t, cfg)
	if parsed.Observability.Tracing.Sampling.Rate != 0 {
		t.Fatalf("tracing.sampling.rate = %v, want 0", parsed.Observability.Tracing.Sampling.Rate)
	}
}
