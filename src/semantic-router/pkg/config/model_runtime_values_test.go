package config

import (
	"strings"
	"testing"
)

func TestModelRuntimeValuesMaterializeEffectiveDefaults(t *testing.T) {
	input := "0.5"
	execution := ModelExecutionSettings{}
	pricing := ModelRuntimePricing{InputCostPerMillionTokens: &input}

	if err := validateModelRuntimeValues("models[worker]", &execution, &pricing); err != nil {
		t.Fatalf("validateModelRuntimeValues() error = %v", err)
	}
	if execution.RequestTimeout != "300s" || execution.StreamTimeout != "300s" {
		t.Fatalf("execution defaults = %+v", execution)
	}
	if pricing.CacheReadCostPerMillionTokens == nil || pricing.CacheWriteCostPerMillionTokens == nil ||
		*pricing.CacheReadCostPerMillionTokens != "0.5" || *pricing.CacheWriteCostPerMillionTokens != "0.5" {
		t.Fatalf("cache price inheritance = %+v", pricing)
	}
	input = "9"
	if *pricing.CacheReadCostPerMillionTokens != "0.5" || *pricing.CacheWriteCostPerMillionTokens != "0.5" {
		t.Fatal("effective cache prices alias the configured input price")
	}
}

func TestCanonicalExportPreservesBillingCurrency(t *testing.T) {
	document := strings.Replace(entrypointRulesYAML, "version: v0.4\n", "version: v0.4\nbilling_currency: USD\n", 1)
	document = strings.Replace(document, "  - name: model-a\n", "  - name: model-a\n    pricing: {input_cost_per_million_tokens: \"0.25\"}\n", 1)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("testAuthoringParser(t).ParseYAMLBytes() error = %v", err)
	}
	if got := CanonicalConfigFromRouterConfig(cfg).BillingCurrency; got != "USD" {
		t.Fatalf("billing currency = %q, want USD", got)
	}
}
