package config

import (
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestProviderReliabilityRoundTripsCanonicalConfig(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: model-a
  models:
    - name: model-a
      reliability:
        lb_policy: least_request
        retry_count: 2
        retry_on: connect-failure,refused-stream
        consecutive_5xx: 5
        base_ejection_time: 45s
        max_ejection_percent: 25
        health_check_path: /health
        health_check_interval: 15s
        health_check_timeout: 3s
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: model-a
  decisions:
    - name: default
      rules:
        operator: AND
      modelRefs:
        - model: model-a
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	reliability := parsed.ModelConfig["model-a"].Reliability
	if reliability.LBPolicy != ProviderLBPolicyLeastRequest ||
		reliability.RetryCount != 2 ||
		reliability.Consecutive5xx != 5 ||
		reliability.BaseEjectionTime != "45s" ||
		reliability.MaxEjectionPercent != 25 ||
		reliability.HealthCheckPath != "/health" {
		t.Fatalf("reliability did not normalize: %#v", reliability)
	}
}

func TestProviderReliabilityTimeoutsAndRetriesRoundTrip(t *testing.T) {
	const doc = `
version: v0.3
providers:
  models:
    - name: model-a
      provider_model_id: model-a
      api_format: openai
      reliability:
        retry_count: 2
        connect_timeout: 3s
        total_timeout: 0s
        idle_timeout: 45s
        per_try_timeout: 20s
        first_byte_timeout: 5s
        retriable_status_codes: [429, 503]
        retry_back_off_base: 50ms
        retry_back_off_max: 1s
        retry_after_max: 30s
        retry_budget_percent: 25
        retry_budget_min_concurrency: 4
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing: {}
`
	want := ProviderReliability{
		RetryCount: 2, ConnectTimeout: "3s", TotalTimeout: "0s", IdleTimeout: "45s", PerTryTimeout: "20s",
		FirstByteTimeout: "5s", RetriableStatusCodes: []int{429, 503}, RetryBackOffBase: "50ms",
		RetryBackOffMax: "1s", RetryAfterMax: "30s", RetryBudgetPercent: 25, RetryBudgetMinConcurrency: 4,
	}
	parsed, err := ParseYAMLBytes([]byte(doc))
	if err != nil {
		t.Fatalf("ParseYAMLBytes: %v", err)
	}
	if got := parsed.ModelConfig["model-a"].Reliability; !reflect.DeepEqual(got, want) {
		t.Fatalf("reliability = %#v, want %#v", got, want)
	}
	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(parsed))
	if err != nil {
		t.Fatal(err)
	}
	reparsed, err := ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("re-import: %v", err)
	}
	if got := reparsed.ModelConfig["model-a"].Reliability; !reflect.DeepEqual(got, want) {
		t.Fatalf("exported reliability = %#v, want %#v", got, want)
	}
}

func TestProviderReliabilityDefaultsRetryOnLikeTheCLI(t *testing.T) {
	if err := validateProviderReliability("model-a", ProviderReliability{RetryCount: 2}); err != nil {
		t.Fatalf("retries without retry_on must take the default %q: %v", DefaultProviderRetryOn, err)
	}
}

func TestProviderReliabilityRejectsInvalidTimeoutsAndRetries(t *testing.T) {
	for name, reliability := range map[string]ProviderReliability{
		"zero connect timeout":   {ConnectTimeout: "0s"},
		"unparsable total":       {TotalTimeout: "soon"},
		"negative idle":          {IdleTimeout: "-1s"},
		"zero per try":           {PerTryTimeout: "0s"},
		"status out of range":    {RetriableStatusCodes: []int{429, 700}},
		"max below base":         {RetryBackOffBase: "100ms", RetryBackOffMax: "50ms"},
		"max below default base": {RetryBackOffMax: "10ms"},
		"budget over 100":        {RetryBudgetPercent: 150},
		"negative concurrency":   {RetryBudgetMinConcurrency: -1},
		"zero retry after max":   {RetryAfterMax: "0s"},
	} {
		if err := validateProviderReliability("model-a", reliability); err == nil {
			t.Errorf("%s: accepted %#v", name, reliability)
		}
	}
}

func TestProviderReliabilityRejectsUnsafeValues(t *testing.T) {
	err := validateProviderReliability("model-a", ProviderReliability{
		LBPolicy:   "random",
		RetryCount: 8,
	})
	if err == nil {
		t.Fatal("invalid reliability config must be rejected")
	}
}
