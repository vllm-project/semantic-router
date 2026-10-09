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

func TestSignalTimeoutAndOnUnscannedValidate(t *testing.T) {
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
	for value, ok := range map[string]bool{"": true, OnErrorAllow: true, OnErrorBlock: true, "skip": false} {
		unscanned := UnscannedConfig{OnUnscanned: value}
		if (unscanned.ValidateOnUnscanned() == nil) != ok {
			t.Fatalf("on_unscanned %q", value)
		}
		if unscanned.UnscannedBlocks() != (value != OnErrorAllow) {
			t.Fatalf("on_unscanned %q blocks", value)
		}
	}
}

func TestSignalTimeoutAndOnUnscannedRoundTripTheCanonicalConfig(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
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
    modules:
      prompt_guard:
        on_unscanned: allow
      classifier:
        pii:
          on_unscanned: block
routing:
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
	if cfg.SignalTimeout() != 2500*time.Millisecond || cfg.PromptGuard.UnscannedBlocks() || !cfg.PIIModel.UnscannedBlocks() {
		t.Fatalf("parsed: timeout %v, guard %q, pii %q", cfg.SignalTimeout(), cfg.PromptGuard.OnUnscanned, cfg.PIIModel.OnUnscanned)
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	if exported.Global == nil || exported.Global.ModelCatalog.SignalTimeoutMs != 2500 || exported.Global.ModelCatalog.Modules.PromptGuard.OnUnscanned != OnErrorAllow {
		t.Fatalf("exported model catalog lost the fields")
	}
}
