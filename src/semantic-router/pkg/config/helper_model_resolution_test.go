package config

import "testing"

func TestResolveModelConfigDuplicateExternalIDIsDeterministic(t *testing.T) {
	cfg := &RouterConfig{
		BackendModels: BackendModels{
			ModelConfig: map[string]ModelParams{
				"z-model": {
					Description:      "z-model",
					ExternalModelIDs: map[string]string{"openai": "shared-provider-id"},
				},
				"a-model": {
					Description:      "a-model",
					ExternalModelIDs: map[string]string{"openai": "shared-provider-id"},
				},
			},
		},
	}

	for i := 0; i < 100; i++ {
		params, ok := cfg.resolveModelConfig("shared-provider-id")
		if !ok {
			t.Fatal("duplicate external model ID was not resolved")
		}
		if params.Description != "a-model" {
			t.Fatalf("resolved owner = %q, want lexically first model a-model", params.Description)
		}
	}
}
