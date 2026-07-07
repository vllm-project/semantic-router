package vllm

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestTargetsFromRouterConfigBuildsVLLMTargets(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"qwen3": {PreferredEndpoints: []string{"primary", "openai"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:       "primary",
					BackendID:  "qwen3-primary",
					EngineKind: "vllm",
					Address:    "127.0.0.1",
					Port:       8000,
					Protocol:   "http",
					APIKey:     "secret",
				},
				{
					Name:                "openai",
					EngineKind:          "",
					ProviderProfileName: "openai-prod",
				},
			},
		},
	}

	targets := TargetsFromRouterConfig(cfg, TargetOptions{MetricsPath: "/metrics"})
	if len(targets) != 1 {
		t.Fatalf("len(targets) = %d, want 1: %#v", len(targets), targets)
	}
	target := targets[0]
	if target.Identity.BackendID != "qwen3-primary" || target.Identity.ModelName != "qwen3" {
		t.Fatalf("target identity = %#v", target.Identity)
	}
	if target.MetricsEndpoint != "http://127.0.0.1:8000/metrics" {
		t.Fatalf("MetricsEndpoint = %q, want http://127.0.0.1:8000/metrics", target.MetricsEndpoint)
	}
	if target.Headers["Authorization"] != "Bearer secret" {
		t.Fatalf("Authorization header = %q, want bearer token", target.Headers["Authorization"])
	}
}

func TestTargetsFromRouterConfigIncludesLegacyLocalVLLMWhenEnabled(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"qwen3": {PreferredEndpoints: []string{"legacy"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "legacy", Address: "::1", Port: 8000, Protocol: "http"},
			},
		},
	}

	targets := TargetsFromRouterConfig(cfg, TargetOptions{IncludeLegacyVLLM: true})
	if len(targets) != 1 {
		t.Fatalf("len(targets) = %d, want 1", len(targets))
	}
	if targets[0].Identity.BackendID != "legacy" {
		t.Fatalf("BackendID = %q, want legacy", targets[0].Identity.BackendID)
	}
	if targets[0].MetricsEndpoint != "http://[::1]:8000/metrics" {
		t.Fatalf("MetricsEndpoint = %q, want IPv6 bracket endpoint", targets[0].MetricsEndpoint)
	}

	targets = TargetsFromRouterConfig(cfg, TargetOptions{})
	if len(targets) != 0 {
		t.Fatalf("len(targets) = %d, want 0 when legacy fallback disabled", len(targets))
	}
}
