package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

// costBaselineDocument has a priced model, an unpriced one, a model priced
// only by the built-in catalog's list price for its provider, and a
// decision's router_replay plugin with the given cost_baseline_model.
func costBaselineDocument(baseline string) []byte {
	return []byte(`version: v0.5
providers:
  defaults:
    model: small
  models:
    - name: small
      backend_refs:
        - endpoint: local:8000
      pricing: {currency: USD, prompt_per_1m: 1, completion_per_1m: 2}
    - name: unpriced
      backend_refs:
        - endpoint: local:8001
    - name: astra
      catalog: openai/gpt-6-astra
      api_format: responses
      backend_refs:
        - name: primary
          provider: openai
recipes:
  - name: default
    decisions:
      - name: route
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: small
        plugins:
          - type: router_replay
            configuration:
              enabled: true
              cost_baseline_model: ` + baseline + `
`)
}

func TestRouterReplayCostBaselineModelMustNameAPricedModel(t *testing.T) {
	for baseline, want := range map[string]string{
		"astra":    "",
		"small":    "",
		"unpriced": `cost_baseline_model: model "unpriced" has no pricing`,
		"missing":  `cost_baseline_model: "missing" is not a configured model`,
	} {
		t.Run(baseline, func(t *testing.T) {
			cfg, err := ParseYAMLBytes(costBaselineDocument(baseline))
			if want == "" {
				if err != nil {
					t.Fatalf("load: %v", err)
				}
				decision := cfg.AllRoutingDecisions()[0]
				if got := cfg.EffectiveRouterReplayConfig(&decision).CostBaselineModel; got != baseline {
					t.Fatalf("effective cost_baseline_model = %q, want %q", got, baseline)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("error = %v, want %q", err, want)
			}
		})
	}
}

func TestModelPricingProvenanceSeparatesListPricesFromConfiguredOnes(t *testing.T) {
	cfg, err := ParseYAMLBytes(costBaselineDocument("small"))
	if err != nil {
		t.Fatal(err)
	}
	for model, want := range map[string]struct {
		provenance PricingProvenance
		priced     bool
	}{
		"small":    {priced: true},
		"astra":    {provenance: PricingProvenance{CatalogListPrice: true}, priced: true},
		"unpriced": {},
		"missing":  {},
	} {
		provenance, priced := cfg.ModelPricingProvenance(model)
		if provenance != want.provenance || priced != want.priced {
			t.Fatalf("%s provenance = %+v, %v; want %+v, %v", model, provenance, priced, want.provenance, want.priced)
		}
		if _, hasRates := cfg.GetFullModelPricing(model); hasRates != priced {
			t.Fatalf("%s provenance priced = %v, but GetFullModelPricing says %v", model, priced, hasRates)
		}
	}
}

func TestModelPricingProvenanceMarksTheCallersLogin(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.5
listeners:
  - name: http
    port: 8899
providers:
  models:
    - name: astra
      catalog: openai/gpt-6-astra
      api_format: responses
      backend_refs:
        - name: primary
          provider: openai
          auth: {credential: caller}
    - name: private-preview
      api_format: responses
      backend_refs:
        - name: primary
          provider: openai
          auth: {credential: caller}
recipes:
  - name: default
`))
	if err != nil {
		t.Fatal(err)
	}
	provenance, priced := cfg.ModelPricingProvenance("astra")
	if !priced || provenance != (PricingProvenance{CatalogListPrice: true, CallerLogin: true}) {
		t.Fatalf("caller provenance = %+v, %v", provenance, priced)
	}
	// A model the catalog does not list has no rates, and is still on the
	// caller's login.
	provenance, priced = cfg.ModelPricingProvenance("private-preview")
	if priced || provenance != (PricingProvenance{CallerLogin: true}) {
		t.Fatalf("unpriced caller provenance = %+v, %v", provenance, priced)
	}
}

func TestModelsOnAClaudeOrChatGPTLoginTakeTheirListPrice(t *testing.T) {
	rate := func(value float64) *float64 { return &value }
	for name, tc := range map[string]struct {
		model string
		want  ModelPricing
	}{
		"claude": {
			model: `catalog: anthropic/claude-sonnet-5
      backend_refs:
        - {name: claude, provider: anthropic, auth: {credential: caller}}`,
			want: ModelPricing{Currency: "USD", PromptPer1M: 2, CompletionPer1M: 10, CachedInputPer1M: 0.2, CacheWritePer1M: rate(2.5), CacheWrite1hPer1M: rate(4)},
		},
		"claude 5.5": {
			model: `catalog: anthropic/claude-sonnet-5.5
      backend_refs:
        - {name: claude, provider: anthropic, auth: {credential: caller}}`,
			want: ModelPricing{Currency: "USD", PromptPer1M: 2, CompletionPer1M: 10, CachedInputPer1M: 0.1, CacheWritePer1M: rate(2.5), CacheWrite1hPer1M: rate(4)},
		},
		"chatgpt": {
			model: `catalog: openai/gpt-5.5
      api_format: responses
      backend_refs:
        - {name: chatgpt, endpoint: "https://chatgpt.com/backend-api/codex", provider: openai, auth: {credential: caller}}`,
			want: ModelPricing{Currency: "USD", PromptPer1M: 5, CompletionPer1M: 30, CachedInputPer1M: 0.5},
		},
		// Without catalog, the backend's provider prices the model it is called with.
		"claude provider_model_id": {
			model: `provider_model_id: claude-sonnet-5-5
      api_format: anthropic
      backend_refs:
        - {name: claude, provider: anthropic, auth: {credential: caller}}`,
			want: ModelPricing{Currency: "USD", PromptPer1M: 2, CompletionPer1M: 10, CachedInputPer1M: 0.1, CacheWritePer1M: rate(2.5), CacheWrite1hPer1M: rate(4)},
		},
		"chatgpt provider_model_id": {
			model: `provider_model_id: gpt-5.5
      api_format: responses
      backend_refs:
        - {name: chatgpt, endpoint: "https://chatgpt.com/backend-api/codex", provider: openai, auth: {credential: caller}}`,
			want: ModelPricing{Currency: "USD", PromptPer1M: 5, CompletionPer1M: 30, CachedInputPer1M: 0.5},
		},
		// A model ID with only a price list entry, no card.
		"chatgpt price list": {
			model: `provider_model_id: gpt-6.1-sol
      api_format: responses
      backend_refs:
        - {name: chatgpt, endpoint: "https://chatgpt.com/backend-api/codex", provider: openai, auth: {credential: caller}}`,
			want: ModelPricing{Currency: "USD", PromptPer1M: 2, CompletionPer1M: 10, CachedInputPer1M: 0.1, CacheWritePer1M: rate(2.5)},
		},
	} {
		cfg, err := ParseYAMLBytes([]byte(`version: v0.5
listeners:
  - name: http
    port: 8899
providers:
  models:
    - name: login
      ` + tc.model + `
recipes:
  - name: default
`))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got, priced := cfg.GetFullModelPricing("login"); !priced || !reflect.DeepEqual(got, tc.want) {
			t.Fatalf("%s pricing = %+v, %v; want %+v", name, got, priced, tc.want)
		}
		if provenance, _ := cfg.ModelPricingProvenance("login"); provenance != (PricingProvenance{CatalogListPrice: true, CallerLogin: true}) {
			t.Fatalf("%s provenance = %+v", name, provenance)
		}
	}
}

// A self-hosted server that serves a vendor's model ID is not billed as the
// vendor: only the backend's own provider prices the model ID.
func TestSelfHostedModelWithAVendorModelIDTakesNoListPrice(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.5
providers:
  models:
    - name: local
      provider_model_id: claude-sonnet-5-5
      backend_refs:
        - {name: local, endpoint: "http://127.0.0.1:8000/v1", provider: vllm}
recipes:
  - name: default
`))
	if err != nil {
		t.Fatal(err)
	}
	if pricing, priced := cfg.GetFullModelPricing("local"); priced {
		t.Fatalf("vllm model priced as the vendor: %+v", pricing)
	}
}

func TestRouterReplayCostBaselineModelLayersFromGlobalPlugins(t *testing.T) {
	document := strings.Replace(string(costBaselineDocument("small")), "              cost_baseline_model: small\n", "", 1) + `global:
  plugins:
    - type: router_replay
      configuration:
        cost_baseline_model: astra
`
	cfg, err := ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	decision := cfg.AllRoutingDecisions()[0]
	if got := cfg.EffectiveRouterReplayConfig(&decision).CostBaselineModel; got != "astra" {
		t.Fatalf("decision inherited cost_baseline_model %q, want astra from global.plugins", got)
	}
	invalid := strings.Replace(document, "cost_baseline_model: astra", "cost_baseline_model: unpriced", 1)
	if _, err := ParseYAMLBytes([]byte(invalid)); err == nil || !strings.Contains(err.Error(), "global.plugins[router_replay].configuration.cost_baseline_model") {
		t.Fatalf("global baseline error = %v, want its global.plugins path", err)
	}
}

func TestRouterReplayCostBaselineModelSurvivesCanonicalExport(t *testing.T) {
	cfg, err := ParseYAMLBytes(costBaselineDocument("astra"))
	if err != nil {
		t.Fatal(err)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("export does not load: %v\n%s", err, exported)
	}
	decision := reloaded.AllRoutingDecisions()[0]
	if got := reloaded.EffectiveRouterReplayConfig(&decision).CostBaselineModel; got != "astra" {
		t.Fatalf("exported cost_baseline_model = %q, want astra", got)
	}
	// The catalog's list price stays a default: the export does not write it
	// as the operator's pricing.
	if provenance, priced := reloaded.ModelPricingProvenance("astra"); !priced || !provenance.CatalogListPrice {
		t.Fatalf("exported astra provenance = %+v, %v", provenance, priced)
	}
}
