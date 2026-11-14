package config

import (
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func TestSignalOverloadPolicyValidates(t *testing.T) {
	for name, test := range map[string]struct {
		policy *SignalOverloadPolicy
		err    string
	}{
		"absent":              {nil, ""},
		"refuse":              {&SignalOverloadPolicy{Action: SignalOverloadRefuse}, ""},
		"retry":               {&SignalOverloadPolicy{Action: SignalOverloadRetry}, ""},
		"retry with its wait": {&SignalOverloadPolicy{Action: SignalOverloadRetry, WaitMs: 60_000, RetryAfterSeconds: 60}, ""},
		"unknown action":      {&SignalOverloadPolicy{Action: "queue"}, "action must be refuse or retry"},
		"negative wait":       {&SignalOverloadPolicy{Action: SignalOverloadRetry, WaitMs: -1}, "wait_ms must be between 0 and 60000"},
		"wait too long":       {&SignalOverloadPolicy{Action: SignalOverloadRetry, WaitMs: 60_001}, "wait_ms must be between 0 and 60000"},
		"retry-after too long": {
			&SignalOverloadPolicy{Action: SignalOverloadRetry, RetryAfterSeconds: 61}, "retry_after_seconds must be between 0 and 60",
		},
		"a wait without retry": {
			&SignalOverloadPolicy{WaitMs: 500}, "apply only to action retry",
		},
		"a retry-after without retry": {
			&SignalOverloadPolicy{Action: SignalOverloadRefuse, RetryAfterSeconds: 2}, "apply only to action retry",
		},
	} {
		err := validateModelSignalOverloadContracts(&RouterConfig{InlineModels: InlineModels{ModelSignalOverload: test.policy}})
		if test.err == "" && err != nil || test.err != "" && (err == nil || !strings.Contains(err.Error(), test.err)) {
			t.Fatalf("%s: error %v, want %q", name, err, test.err)
		}
	}
}

// Without the policy, or with refuse, a request refused because its guard
// failed closed on an overloaded runtime stays refused, with no wait. With
// retry, it waits wait_ms and gets a Retry-After of that, in whole seconds,
// unless retry_after_seconds says otherwise.
func TestSignalOverloadPolicyDefaults(t *testing.T) {
	for _, test := range []struct {
		policy     *SignalOverloadPolicy
		retries    bool
		wait       time.Duration
		retryAfter int
	}{
		{nil, false, 0, 1},
		{&SignalOverloadPolicy{Action: SignalOverloadRefuse}, false, 0, 1},
		{&SignalOverloadPolicy{Action: SignalOverloadRetry}, true, 0, 1},
		{&SignalOverloadPolicy{Action: SignalOverloadRetry, WaitMs: 1_500}, true, 1500 * time.Millisecond, 2},
		{&SignalOverloadPolicy{Action: SignalOverloadRetry, WaitMs: 1_500, RetryAfterSeconds: 10}, true, 1500 * time.Millisecond, 10},
	} {
		cfg := &RouterConfig{InlineModels: InlineModels{ModelSignalOverload: test.policy}}
		if cfg.SignalOverloadRetries() != test.retries || cfg.SignalOverloadWait() != test.wait || cfg.SignalOverloadRetryAfter() != test.retryAfter {
			t.Fatalf("%+v: retries %v, wait %v, retry-after %d", test.policy, cfg.SignalOverloadRetries(), cfg.SignalOverloadWait(), cfg.SignalOverloadRetryAfter())
		}
	}
}

func TestSignalOverloadPolicyRoundTripsTheCanonicalConfig(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - name: b
          endpoint: localhost:8000
          protocol: http
global:
  model_catalog:
    on_signal_overload:
      action: retry
      wait_ms: 2000
      retry_after_seconds: 3
recipes:
  - name: default
    decisions:
      - name: d
        priority: 1
        rules: {operator: AND, conditions: [{type: keyword, name: k}]}
        modelRefs: [{model: m}]
    signals:
      keywords:
        - name: k
          operator: OR
          keywords: [hello]
`))
	if err != nil {
		t.Fatal(err)
	}
	want := SignalOverloadPolicy{Action: SignalOverloadRetry, WaitMs: 2000, RetryAfterSeconds: 3}
	if cfg.ModelSignalOverload == nil || *cfg.ModelSignalOverload != want {
		t.Fatalf("parsed %+v, want %+v", cfg.ModelSignalOverload, want)
	}
	exported, err := yaml.Marshal(canonicalModelCatalogFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(exported), "on_signal_overload:\n    action: retry\n    wait_ms: 2000\n    retry_after_seconds: 3\n") {
		t.Fatalf("the exported catalog lost the policy:\n%s", exported)
	}
}
