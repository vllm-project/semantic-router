package config

import (
	"math"
	"strings"
	"testing"
)

func TestValidateModelPricingContracts(t *testing.T) {
	cacheWrite := 0.625
	tests := []struct {
		name    string
		pricing ModelPricing
		wantErr string
	}{
		{
			name: "complete pricing",
			pricing: ModelPricing{
				Currency: "USD", PromptPer1M: 0.5, CompletionPer1M: 1.5,
				CachedInputPer1M: 0.05, CacheWritePer1M: &cacheWrite,
			},
		},
		{name: "currency omitted defaults at lookup", pricing: ModelPricing{PromptPer1M: 0.5}},
		{name: "explicit free model", pricing: ModelPricing{Currency: "USD"}},
		{name: "invalid currency", pricing: ModelPricing{Currency: "usd"}, wantErr: "pricing.currency"},
		{name: "negative input", pricing: ModelPricing{PromptPer1M: -0.1}, wantErr: "pricing.prompt_per_1m"},
		{name: "infinite output", pricing: ModelPricing{CompletionPer1M: math.Inf(1)}, wantErr: "pricing.completion_per_1m"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
				"model-a": {Pricing: test.pricing},
			}}}
			err := validateModelPricingContracts(cfg)
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("validate pricing: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestParseYAMLBytesRejectsInvalidProviderPricing(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: model-a
      pricing:
        currency: USD
        prompt_per_1m: -0.01
routing:
  modelCards:
    - name: model-a
`))
	if err == nil || !strings.Contains(err.Error(), "providers.models[model-a].pricing.prompt_per_1m") {
		t.Fatalf("error = %v, want provider pricing path", err)
	}
}
