package config

import "testing"

func TestGetMostExpensiveFullModelPricingIncludesCacheWriteRate(t *testing.T) {
	premiumWriteRate := 20.0
	config := &RouterConfig{
		BackendModels: BackendModels{
			ModelConfig: map[string]ModelParams{
				"high-output": {
					Pricing: ModelPricing{
						PromptPer1M:     5,
						CompletionPer1M: 30,
					},
				},
				"high-cache-write": {
					Pricing: ModelPricing{
						PromptPer1M:     2.5,
						CacheWritePer1M: &premiumWriteRate,
						CompletionPer1M: 20,
					},
				},
			},
		},
	}

	model, pricing, ok := config.GetMostExpensiveFullModelPricing()
	if !ok {
		t.Fatal("expected configured pricing")
	}
	if model != "high-cache-write" || pricing.CacheWritePer1M == nil || *pricing.CacheWritePer1M != premiumWriteRate {
		t.Fatalf("expected premium cache-write model, got model=%q pricing=%#v", model, pricing)
	}
}

func TestGetFullModelPricingPreservesExplicitFreeCacheWrites(t *testing.T) {
	freeWriteRate := 0.0
	config := &RouterConfig{
		BackendModels: BackendModels{
			ModelConfig: map[string]ModelParams{
				"free-writes": {
					Pricing: ModelPricing{CacheWritePer1M: &freeWriteRate},
				},
			},
		},
	}

	pricing, ok := config.GetFullModelPricing("free-writes")
	if !ok || pricing.CacheWritePer1M == nil || *pricing.CacheWritePer1M != 0 {
		t.Fatalf("expected explicit zero cache-write rate, got pricing=%#v configured=%v", pricing, ok)
	}
}
