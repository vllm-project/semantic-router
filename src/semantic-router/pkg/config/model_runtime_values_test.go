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

func TestCanonicalExportPreservesGlobalBillingCurrency(t *testing.T) {
	document := strings.Replace(entrypointRulesYAML, "global:\n", "global:\n  billing:\n    currency: USD\n", 1)
	document = strings.Replace(document, "  - name: model-a\n", "  - name: model-a\n    pricing: {input_cost_per_million_tokens: \"0.25\"}\n", 1)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("testAuthoringParser(t).ParseYAMLBytes() error = %v", err)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if exported.Global == nil || exported.Global.Billing == nil || exported.Global.Billing.Currency != "USD" {
		got := ""
		if exported.Global != nil && exported.Global.Billing != nil {
			got = exported.Global.Billing.Currency
		}
		t.Fatalf("billing currency = %q, want USD", got)
	}
}

func TestCanonicalBillingCurrencyOwnership(t *testing.T) {
	priced := strings.Replace(
		entrypointRulesYAML,
		"  - name: model-a\n",
		"  - name: model-a\n    pricing: {input_cost_per_million_tokens: \"0.25\"}\n",
		1,
	)

	if _, err := testAuthoringParser(t).ParseYAMLBytes([]byte(priced)); err == nil ||
		!strings.Contains(err.Error(), "global.billing.currency is required") {
		t.Fatalf("missing currency error = %v", err)
	}
	empty := strings.Replace(priced, "global:\n", "global:\n  billing: {}\n", 1)
	if _, err := testAuthoringParser(t).ParseYAMLBytes([]byte(empty)); err == nil ||
		!strings.Contains(err.Error(), "global.billing.currency is required when global.billing is configured") {
		t.Fatalf("empty billing error = %v", err)
	}

	invalid := strings.Replace(priced, "global:\n", "global:\n  billing:\n    currency: usd\n", 1)
	if _, err := testAuthoringParser(t).ParseYAMLBytes([]byte(invalid)); err == nil ||
		!strings.Contains(err.Error(), "global.billing.currency must be an uppercase ISO-4217 code") {
		t.Fatalf("invalid currency error = %v", err)
	}

	_, err := ParseYAMLBytes([]byte(`
version: v0.4
global:
  control_plane: {mode: managed}
  billing: {currency: USD}
`))
	if err == nil ||
		!strings.Contains(err.Error(), "managed mode takes currency from Namespace") {
		t.Fatalf("managed currency error = %v", err)
	}
}
