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

func TestReconcilePreservesTracingSamplingRateZero(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.Observability.Tracing.Sampling.Rate = 0

	parsed := reconcileRoundTrip(t, cfg)
	if parsed.Observability.Tracing.Sampling.Rate != 0 {
		t.Fatalf("tracing.sampling.rate = %v, want 0", parsed.Observability.Tracing.Sampling.Rate)
	}
}

func TestReconcilePreservesNLIFilteringDisabled(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.HallucinationMitigation.HallucinationModel.EnableNLIFiltering = false

	parsed := reconcileRoundTrip(t, cfg)
	if parsed.HallucinationMitigation.HallucinationModel.EnableNLIFiltering {
		t.Fatal("hallucination detector enable_nli_filtering = true, want false")
	}
}
