package config

import (
	"strings"
	"testing"
)

func TestSelectBestEndpointWithDetailsSkipsUnresolvableHighWeightEndpoint(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{
			ModelConfig: map[string]ModelParams{
				"agent-model": {PreferredEndpoints: []string{"bad-primary", "healthy-fallback"}},
			},
			VLLMEndpoints: []VLLMEndpoint{
				{
					Name:                "bad-primary",
					Weight:              100,
					ProviderProfileName: "missing-profile",
				},
				{
					Name:    "healthy-fallback",
					Address: "127.0.0.1",
					Port:    8000,
					Weight:  1,
				},
			},
		},
	}

	address, endpointName, found, err := cfg.SelectBestEndpointWithDetailsForModel("agent-model")
	if err != nil {
		t.Fatalf("expected endpoint fallback, got error: %v", err)
	}
	if !found {
		t.Fatal("expected fallback endpoint to resolve")
	}
	if endpointName != "healthy-fallback" || address != "127.0.0.1:8000" {
		t.Fatalf("unexpected endpoint fallback: endpoint=%q address=%q", endpointName, address)
	}
}

func TestSelectBestEndpointWithDetailsReportsAllResolutionFailures(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{
			ModelConfig: map[string]ModelParams{
				"agent-model": {PreferredEndpoints: []string{"bad-primary", "bad-fallback"}},
			},
			VLLMEndpoints: []VLLMEndpoint{
				{
					Name:                "bad-primary",
					Weight:              10,
					ProviderProfileName: "missing-primary",
				},
				{
					Name:                "bad-fallback",
					Weight:              1,
					ProviderProfileName: "missing-fallback",
				},
			},
		},
	}

	_, _, found, err := cfg.SelectBestEndpointWithDetailsForModel("agent-model")
	if found {
		t.Fatal("expected no endpoint to resolve")
	}
	if err == nil {
		t.Fatal("expected aggregated endpoint resolution error")
	}
	message := err.Error()
	for _, want := range []string{"bad-primary", "missing-primary", "bad-fallback", "missing-fallback"} {
		if !strings.Contains(message, want) {
			t.Fatalf("expected error to contain %q, got %q", want, message)
		}
	}
}
