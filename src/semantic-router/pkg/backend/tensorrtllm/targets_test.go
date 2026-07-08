package tensorrtllm

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestTargetsFromRouterConfigTRTLLM(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"llama-3.1-70b": {
					PreferredEndpoints: []string{"trtllm-a"},
				},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:       "trtllm-a",
					BackendID:  "trtllm-a",
					EngineKind: "tensorrt-llm",
					Address:    "10.0.0.5",
					Port:       8000,
					Model:      "llama-3.1-70b",
					Protocol:   "http",
				},
			},
		},
	}
	targets := TargetsFromRouterConfig(cfg, TargetOptions{})
	if len(targets) != 1 {
		t.Fatalf("expected 1 target, got %d", len(targets))
	}
	tgt := targets[0]
	// Default Triton metrics port 8002, not the inference port 8000.
	if tgt.MetricsEndpoint != "http://10.0.0.5:8002/metrics" {
		t.Errorf("MetricsEndpoint = %q, want http://10.0.0.5:8002/metrics", tgt.MetricsEndpoint)
	}
	if tgt.Identity.EngineKind != "tensorrt-llm" {
		t.Errorf("EngineKind = %q", tgt.Identity.EngineKind)
	}
	if tgt.Identity.ReplicaID != "10.0.0.5:8000" {
		t.Errorf("ReplicaID = %q, want 10.0.0.5:8000", tgt.Identity.ReplicaID)
	}
}

func TestTargetsRespectExplicitMetricsPortAndPath(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"m": {PreferredEndpoints: []string{"trtllm-b"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{
					Name:        "trtllm-b",
					BackendID:   "trtllm-b",
					EngineKind:  "tensorrt-llm",
					Address:     "host",
					Port:        9000,
					MetricsPort: 9100,
					MetricsPath: "custom/metrics",
					Model:       "m",
					Protocol:    "http",
				},
			},
		},
	}
	targets := TargetsFromRouterConfig(cfg, TargetOptions{})
	if len(targets) != 1 {
		t.Fatalf("expected 1 target, got %d", len(targets))
	}
	if targets[0].MetricsEndpoint != "http://host:9100/custom/metrics" {
		t.Errorf("MetricsEndpoint = %q, want http://host:9100/custom/metrics", targets[0].MetricsEndpoint)
	}
}

func TestTargetsSkipNonTRTLLMEndpoints(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"m": {PreferredEndpoints: []string{"vllm-a", "trtllm-a"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{
				{Name: "vllm-a", BackendID: "vllm-a", EngineKind: "vllm", Address: "h1", Port: 8000, Model: "m", Protocol: "http"},
				{Name: "trtllm-a", BackendID: "trtllm-a", EngineKind: "tensorrt-llm", Address: "h2", Port: 8000, Model: "m", Protocol: "http"},
			},
		},
	}
	targets := TargetsFromRouterConfig(cfg, TargetOptions{})
	if len(targets) != 1 {
		t.Fatalf("expected 1 TRT-LLM target (vLLM skipped), got %d", len(targets))
	}
	if targets[0].Identity.BackendID != "trtllm-a" {
		t.Errorf("BackendID = %q, want trtllm-a", targets[0].Identity.BackendID)
	}
}

func TestTargetsDedupe(t *testing.T) {
	ep := config.VLLMEndpoint{Name: "trtllm-a", BackendID: "trtllm-a", EngineKind: "tensorrt-llm", Address: "h", Port: 8000, Model: "m", Protocol: "http"}
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"m": {PreferredEndpoints: []string{"trtllm-a", "trtllm-a"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{ep, ep},
		},
	}
	targets := TargetsFromRouterConfig(cfg, TargetOptions{})
	if len(targets) != 1 {
		t.Fatalf("expected 1 deduped target, got %d", len(targets))
	}
}

func TestTargetsNilConfig(t *testing.T) {
	if got := TargetsFromRouterConfig(nil, TargetOptions{}); got != nil {
		t.Errorf("expected nil for nil config, got %v", got)
	}
}
