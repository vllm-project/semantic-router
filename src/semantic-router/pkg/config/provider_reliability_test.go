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
        request_timeout: 120s
        stream_idle_timeout: 30s
        connect_timeout: 5s
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
		reliability.HealthCheckPath != "/health" ||
		reliability.RequestTimeout != "120s" ||
		reliability.StreamIdleTimeout != "30s" ||
		reliability.ConnectTimeout != "5s" {
		t.Fatalf("reliability did not normalize: %#v", reliability)
	}
}

func TestProviderReliabilityRejectsUnsafeValues(t *testing.T) {
	tests := []struct {
		name        string
		reliability ProviderReliability
	}{
		{
			name: "invalid lb policy",
			reliability: ProviderReliability{
				LBPolicy:   "random",
				RetryCount: 2,
				RetryOn:    "connect-failure",
			},
		},
		{
			name: "retry count too high",
			reliability: ProviderReliability{
				RetryCount: 8,
			},
		},
		{
			name: "retry enabled without retry_on",
			reliability: ProviderReliability{
				RetryCount: 2,
			},
		},
		{
			name: "negative consecutive 5xx",
			reliability: ProviderReliability{
				Consecutive5xx: -1,
			},
		},
		{
			name: "invalid base ejection time duration",
			reliability: ProviderReliability{
				BaseEjectionTime: "invalid-duration",
			},
		},
		{
			name: "invalid health check timeout",
			reliability: ProviderReliability{
				HealthCheckTimeout: "bad",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateProviderReliability("model-a", tc.reliability)
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
		})
	}
}

func TestProviderReliabilityRejectsUnsafeTimeouts(t *testing.T) {
	tests := []struct {
		name        string
		reliability ProviderReliability
	}{
		{
			name: "invalid request timeout format",
			reliability: ProviderReliability{
				RequestTimeout: "abc",
			},
		},
		{
			name: "negative request timeout",
			reliability: ProviderReliability{
				RequestTimeout: "-10s",
			},
		},
		{
			name: "zero request timeout without stream idle timeout",
			reliability: ProviderReliability{
				RequestTimeout: "0s",
			},
		},
		{
			name: "zero request timeout with zero stream idle timeout",
			reliability: ProviderReliability{
				RequestTimeout:    "0s",
				StreamIdleTimeout: "0s",
			},
		},
		{
			name: "negative stream idle timeout",
			reliability: ProviderReliability{
				StreamIdleTimeout: "-5s",
			},
		},
		{
			name: "invalid stream idle timeout format",
			reliability: ProviderReliability{
				StreamIdleTimeout: "not-a-duration",
			},
		},
		{
			name: "zero connect timeout",
			reliability: ProviderReliability{
				ConnectTimeout: "0s",
			},
		},
		{
			name: "negative connect timeout",
			reliability: ProviderReliability{
				ConnectTimeout: "-1s",
			},
		},
		{
			name: "invalid connect timeout format",
			reliability: ProviderReliability{
				ConnectTimeout: "xyz",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateProviderReliability("model-a", tc.reliability)
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
		})
	}
}

func TestProviderReliabilityAllowsZeroRequestTimeoutWithStreamIdleTimeout(t *testing.T) {
	rel := ProviderReliability{
		RequestTimeout:    "0s",
		StreamIdleTimeout: "30s",
	}
	if err := validateProviderReliability("model-a", rel); err != nil {
		t.Fatalf("expected valid 0s request_timeout with positive stream_idle_timeout, got: %v", err)
	}
}
