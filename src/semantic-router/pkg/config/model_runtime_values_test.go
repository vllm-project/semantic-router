package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestModelControlMaterializesEffectiveDefaults(t *testing.T) {
	input := "0.5"
	execution, pricing, err := compileModelControl(
		"worker", ModelControl{}, ModelRuntimePricing{InputCostPerMillionTokens: &input},
	)
	if err != nil {
		t.Fatalf("compileModelControl() error = %v", err)
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

func TestModelControlErrorsUsePublicNestedPaths(t *testing.T) {
	_, _, err := compileModelControl(
		"worker",
		ModelControl{Timeout: &ModelTimeout{Request: "0s"}},
		ModelRuntimePricing{},
	)
	if err == nil || !strings.Contains(err.Error(), "providers.models[worker].control.timeout.request") {
		t.Fatalf("nested timeout error = %v", err)
	}
}

func TestModelControlRejectsExplicitEmptyTimeout(t *testing.T) {
	document := strings.Replace(
		strictV03AuthoringYAML,
		"    - name: model-a\n",
		"    - name: model-a\n      control:\n        timeout: {request: \"\"}\n",
		1,
	)
	_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), "providers.models[0].control.timeout.request") {
		t.Fatalf("empty timeout error = %v", err)
	}
}

func TestModelPricingRequiresQuotedDecimalStrings(t *testing.T) {
	document := strings.Replace(
		humanAuthoringFixture,
		`input_cost_per_million_tokens: "0.5"`,
		"input_cost_per_million_tokens: 0.5",
		1,
	)
	_, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), "must be a quoted decimal string") {
		t.Fatalf("numeric price error = %v", err)
	}
}

func TestCanonicalExportPreservesGlobalBillingCurrency(t *testing.T) {
	document := strings.Replace(strictV03AuthoringYAML, "global:\n", "global:\n  billing:\n    currency: USD\n", 1)
	document = strings.Replace(document, "    - name: model-a\n", "    - name: model-a\n      pricing: {input_cost_per_million_tokens: \"0.25\"}\n", 1)
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
		strictV03AuthoringYAML,
		"    - name: model-a\n",
		"    - name: model-a\n      pricing: {input_cost_per_million_tokens: \"0.25\"}\n",
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
}

func TestPublicModelDefaultsCompileWithoutExpandingSparseAuthoring(t *testing.T) {
	document := strings.Replace(
		strictV03AuthoringYAML,
		"    - name: model-a\n",
		"    - name: model-a\n      pricing: {input_cost_per_million_tokens: \"0.500\"}\n",
		1,
	)
	document = strings.Replace(document, "global:\n", "global:\n  billing: {currency: USD}\n", 1)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	model := cfg.ModelConfig["model-a"]
	if model.Execution.RequestTimeout != "300s" || model.Execution.StreamTimeout != "300s" {
		t.Fatalf("effective execution defaults = %+v", model.Execution)
	}
	if model.RuntimePricing.CacheReadCostPerMillionTokens == nil ||
		*model.RuntimePricing.CacheReadCostPerMillionTokens != "0.5" {
		t.Fatalf("effective pricing = %+v", model.RuntimePricing)
	}

	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	text := string(exported)
	if strings.Contains(text, "request: 300s") || strings.Contains(text, "stream: 300s") ||
		strings.Contains(text, "cache_read_cost_per_million_tokens") ||
		strings.Contains(text, "cache_write_cost_per_million_tokens") {
		t.Fatalf("effective defaults leaked into sparse public YAML:\n%s", text)
	}
	public := CanonicalConfigFromRouterConfig(cfg)
	price := public.Providers.Models[0].Pricing.InputCostPerMillionTokens
	if price == nil || *price != "0.500" {
		t.Fatalf("public price spelling did not round trip: %+v", public.Providers.Models[0].Pricing)
	}
}

func TestPublicRetryDefaultIsEffectiveOnly(t *testing.T) {
	document := strings.Replace(
		strictV03AuthoringYAML,
		"    - name: model-a\n",
		"    - name: model-a\n      control:\n        retry: {count: 2}\n",
		1,
	)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	model := cfg.ModelConfig["model-a"]
	if len(model.Execution.RetryOn) != 1 || model.Execution.RetryOn[0] != ModelRetryUnavailable {
		t.Fatalf("effective retry evidence = %v", model.Execution.RetryOn)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if exported.Providers.Models[0].Control.Retry == nil ||
		len(exported.Providers.Models[0].Control.Retry.On) != 0 {
		t.Fatalf("retry default leaked into public source: %+v", exported.Providers.Models[0].Control)
	}
}
