package config

import (
	"strings"
	"testing"
	"time"
)

func TestDecisionDeploymentsTakeAScanBudgetAndNeverTruncate(t *testing.T) {
	for name, tc := range map[string]struct {
		input ModelInputBudget
		ok    bool
		scan  int
	}{
		"no input":                {ModelInputBudget{}, true, 0},
		"a scan budget":           {ModelInputBudget{MaxTokens: 65536, Overflow: "window"}, true, 65536},
		"truncate":                {ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}, false, 0},
		"a reject budget":         {ModelInputBudget{MaxTokens: 512, Overflow: "reject"}, false, 0},
		"window without a budget": {ModelInputBudget{Overflow: "window"}, false, 0},
	} {
		deployment := ModelDeployment{Provider: ModelRuntimeProvider, Input: tc.input}
		err := deployment.ValidateDecisionInput("vela2")
		if tc.ok != (err == nil) || (err != nil && !strings.Contains(err.Error(), "never truncate")) {
			t.Fatalf("%s: %v", name, err)
		}
		if got := deployment.ScanBudget(); tc.ok && got != tc.scan {
			t.Fatalf("%s: scan budget %d", name, got)
		}
	}
}

func TestSignalTimeoutValidates(t *testing.T) {
	cfg := &RouterConfig{}
	for _, ms := range []int{0, 300, maxSignalTimeoutMs} {
		cfg.ModelSignalTimeoutMs = ms
		if err := validateModelSignalTimeoutContracts(cfg); err != nil {
			t.Fatalf("%d: %v", ms, err)
		}
	}
	for _, ms := range []int{-1, maxSignalTimeoutMs + 1} {
		cfg.ModelSignalTimeoutMs = ms
		if validateModelSignalTimeoutContracts(cfg) == nil {
			t.Fatalf("%d must be refused", ms)
		}
	}
}

func TestSignalTimeoutRoundTripsTheCanonicalConfig(t *testing.T) {
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
    signal_timeout_ms: 2500
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
	if cfg.SignalTimeout() != 2500*time.Millisecond {
		t.Fatalf("parsed timeout %v", cfg.SignalTimeout())
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if exported.Global == nil || exported.Global.ModelCatalog.SignalTimeoutMs != 2500 {
		t.Fatalf("exported model catalog lost the signal timeout")
	}
}
