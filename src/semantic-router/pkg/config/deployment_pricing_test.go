package config

import (
	"math"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestDeploymentPricing(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"unpriced":  {Provider: ModelRuntimeProvider},
		"priced":    {Pricing: &ModelPricing{PromptPer1M: 2, CompletionPer1M: 4}},
		"free":      {Pricing: &ModelPricing{}},
		"inherited": {Model: "provider-model"},
	}
	cfg.ModelConfig = map[string]ModelParams{"provider-model": {Pricing: ModelPricing{PromptPer1M: 3, Currency: "EUR"}}}
	for _, tc := range []struct {
		name     string
		priced   bool
		input    float64
		currency string
	}{
		{"unpriced", false, 0, ""}, {"missing", false, 0, ""}, {"priced", true, 2, "USD"}, {"free", true, 0, "USD"}, {"inherited", true, 3, "EUR"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, ok := cfg.DeploymentPricing(tc.name)
			if ok != tc.priced || p.PromptPer1M != tc.input || p.Currency != tc.currency {
				t.Fatalf("pricing = %+v %v", p, ok)
			}
		})
	}
}

func TestDeploymentPricingValidationAndYAML(t *testing.T) {
	for _, pricing := range []ModelPricing{{PromptPer1M: -1}, {CompletionPer1M: math.Inf(1)}, {Currency: "usd"}} {
		if err := (ModelDeployment{Pricing: &pricing}).validate(); err == nil {
			t.Fatalf("accepted %+v", pricing)
		}
	}
	original := ModelDeployment{Provider: ModelRuntimeProvider, Pricing: &ModelPricing{PromptPer1M: 2}}
	data, err := yaml.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ModelDeployment
	if err = yaml.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Pricing == nil || *decoded.Pricing != *original.Pricing {
		t.Fatalf("roundtrip = %+v", decoded)
	}
}

func TestDeploymentPricingCanonicalRoundtrip(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.5
recipes: [{name: default}]
global:
  model_catalog:
    deployments:
      priced:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        pricing: {currency: USD, prompt_per_1m: 2, completion_per_1m: 4, cache_write_per_1m: 0}
      free:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        pricing: {}
`))
	if err != nil {
		t.Fatal(err)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	data, err := yaml.Marshal(exported)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := ParseYAMLBytes(data)
	if err != nil {
		t.Fatal(err)
	}
	pricing, ok := reloaded.DeploymentPricing("priced")
	if !ok || pricing.PromptPer1M != 2 || pricing.CompletionPer1M != 4 || pricing.CacheWritePer1M == nil {
		t.Fatalf("pricing = %+v", pricing)
	}
	pricing, ok = reloaded.DeploymentPricing("free")
	if !ok || pricing.Currency != "USD" {
		t.Fatalf("free = %+v", pricing)
	}
	exported.Global.ModelCatalog.Deployments["priced"].Pricing.PromptPer1M = 100
	if pricing, _ := cfg.DeploymentPricing("priced"); pricing.PromptPer1M != 2 {
		t.Fatal("export changed runtime pricing")
	}
}
