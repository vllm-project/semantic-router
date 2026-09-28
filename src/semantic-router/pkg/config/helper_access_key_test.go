package config

import "testing"

func TestGetModelAccessKeyForProviderDoesNotCrossProviderBoundary(t *testing.T) {
	t.Parallel()

	config := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
		"frontier": {
			AccessKey:  "provider-a-secret",
			AccessKeys: map[string]string{"provider-a": "provider-a-secret"},
		},
	}}}

	if got := config.GetModelAccessKeyForProvider("frontier", "provider-a"); got != "provider-a-secret" {
		t.Fatalf("provider-a key = %q, want provider-a-secret", got)
	}
	if got := config.GetModelAccessKeyForProvider("frontier", "provider-b"); got != "" {
		t.Fatalf("provider-b inherited another provider's key %q", got)
	}
	if got := config.GetModelAccessKey("frontier"); got != "provider-a-secret" {
		t.Fatalf("legacy provider-neutral key = %q, want provider-a-secret", got)
	}
}

func TestGetModelAccessKeyForProviderKeepsLoRAProviderIsolation(t *testing.T) {
	t.Parallel()

	config := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
		"base": {
			AccessKey:  "provider-a-secret",
			AccessKeys: map[string]string{"provider-a": "provider-a-secret"},
			LoRAs:      []LoRAAdapter{{Name: "adapter"}},
		},
	}}}

	if got := config.GetModelAccessKeyForProvider("adapter", "provider-a"); got != "provider-a-secret" {
		t.Fatalf("LoRA provider-a key = %q, want provider-a-secret", got)
	}
	if got := config.GetModelAccessKeyForProvider("adapter", "provider-b"); got != "" {
		t.Fatalf("LoRA provider-b inherited another provider's key %q", got)
	}
	if got := config.GetModelAccessKey("adapter"); got != "provider-a-secret" {
		t.Fatalf("legacy provider-neutral LoRA key = %q, want provider-a-secret", got)
	}
}
