package config

import (
	"reflect"
	"testing"
)

const externalGatewayResponsesValues = "deploy/kubernetes/ai-gateway/semantic-router-values/responses-state.yaml"

func TestExternalGatewayResponsesProfileIsStateOnly(t *testing.T) {
	cfg, err := ParseYAMLBytes(readValuesConfigAsset(t, externalGatewayResponsesValues))
	if err != nil {
		t.Fatalf("parse external gateway Responses profile: %v", err)
	}

	if cfg.ModelSelection.Enabled {
		t.Fatal("external gateway Responses profile must not enable model selection")
	}
	if !cfg.ResponseAPI.Enabled {
		t.Fatal("external gateway Responses profile must enable Responses state")
	}
	if cfg.ResponseAPI.StoreBackend != "memory" {
		t.Fatalf("response store backend = %q, want memory for the minimal profile", cfg.ResponseAPI.StoreBackend)
	}
	if len(cfg.Listeners) != 0 {
		t.Fatalf("external gateway owns public listeners, got %d Semantic Router listeners", len(cfg.Listeners))
	}
	if len(cfg.VLLMEndpoints) != 0 || len(cfg.ProviderProfiles) != 0 {
		t.Fatalf("external gateway owns backends and credentials, got endpoints=%d provider_profiles=%d", len(cfg.VLLMEndpoints), len(cfg.ProviderProfiles))
	}
	if cfg.HasRoutingDecisions() || !reflect.DeepEqual(cfg.Signals, Signals{}) {
		t.Fatalf("state-only profile must not declare decisions or routing signals: decisions=%d signals=%+v", len(cfg.AllRoutingDecisions()), cfg.Signals)
	}

	assertExternalGatewayModelMetadata(t, cfg)
}

func assertExternalGatewayModelMetadata(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	const model = "openai/gpt-oss-20b"
	if len(cfg.ModelConfig) != 1 {
		t.Fatalf("model metadata entries = %d, want 1", len(cfg.ModelConfig))
	}
	if got := cfg.GetModelAPIFormat(model); got != APIFormatOpenAI {
		t.Fatalf("model API format = %q, want %q", got, APIFormatOpenAI)
	}
	if endpoints := cfg.GetEndpointsForModel(model); len(endpoints) != 0 {
		t.Fatalf("metadata-only model resolved %d Semantic Router endpoints", len(endpoints))
	}
}
