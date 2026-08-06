package config

import "testing"

func TestProviderReliabilityRoundTripsCanonicalConfig(t *testing.T) {
	parsed, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    default_model: model-a
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

func TestProviderReliabilityRejectsUnsafeValues(t *testing.T) {
	err := validateProviderReliability("model-a", ProviderReliability{
		LBPolicy:   "random",
		RetryCount: 8,
	})
	if err == nil {
		t.Fatal("invalid reliability config must be rejected")
	}
}
