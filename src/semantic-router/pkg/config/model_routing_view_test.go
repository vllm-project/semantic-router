package config

import (
	"strings"
	"testing"
)

func TestRoutingViewAndScanWindowDefaultsByProvider(t *testing.T) {
	remote := ModelDeployment{Provider: SystemOneProvider, Endpoint: "https://decisions.example.com/v1/systemone", Model: "d3"}
	if tokens, share := remote.RoutingViewTokens(); tokens != 4096 || share != 0.25 {
		t.Fatalf("a systemone deployment's view: %d tokens, head %.2f", tokens, share)
	}
	if tokens, overlap := remote.ScanWindowTokens(); tokens != 4096 || overlap != 256 {
		t.Fatalf("a systemone deployment's windows: %d tokens overlapping %d", tokens, overlap)
	}
	local := ModelDeployment{Provider: ModelRuntimeProvider}
	if tokens, share := local.RoutingViewTokens(); tokens != 8192 || share != 0.25 {
		t.Fatalf("a model_runtime deployment's view: %d tokens, head %.2f", tokens, share)
	}
	if tokens, overlap := local.ScanWindowTokens(); tokens != 8192 || overlap != 512 {
		t.Fatalf("a model_runtime deployment's windows: %d tokens overlapping %d", tokens, overlap)
	}
	half, none := 0.5, 0
	local.RoutingView = &DeploymentRoutingView{MaxTokens: 2048, HeadShare: &half}
	local.ScanWindow = &DeploymentScanWindow{Tokens: 1024, Overlap: &none}
	if tokens, share := local.RoutingViewTokens(); tokens != 2048 || share != 0.5 {
		t.Fatalf("a deployment's own view wins: %d tokens, head %.2f", tokens, share)
	}
	if tokens, overlap := local.ScanWindowTokens(); tokens != 1024 || overlap != 0 {
		t.Fatalf("a deployment's own windows win: %d tokens overlapping %d", tokens, overlap)
	}
	if floor := local.ScanWindowRereadFloor(); floor != 0.9 {
		t.Fatalf("the default reread floor is 0.9, got %v", floor)
	}
	local.ScanWindow.RereadFloor = &half
	if floor := local.ScanWindowRereadFloor(); floor != 0.5 {
		t.Fatalf("a deployment's own reread floor wins: %v", floor)
	}
}

func TestRoutingViewAndScanWindowBoundsAreValidated(t *testing.T) {
	share := func(value float64) *float64 { return &value }
	overlap := func(value int) *int { return &value }
	for name, tc := range map[string]struct {
		deployment ModelDeployment
		err        string
	}{
		"systemone":      {ModelDeployment{Provider: SystemOneProvider, RoutingView: &DeploymentRoutingView{MaxTokens: 4096}, ScanWindow: &DeploymentScanWindow{Tokens: 4096}}, ""},
		"largest":        {ModelDeployment{Provider: ModelRuntimeProvider, RoutingView: &DeploymentRoutingView{MaxTokens: MaxModelInputTokens}}, ""},
		"head only":      {ModelDeployment{Provider: ModelRuntimeProvider, RoutingView: &DeploymentRoutingView{HeadShare: share(1)}}, ""},
		"too small":      {ModelDeployment{Provider: ModelRuntimeProvider, RoutingView: &DeploymentRoutingView{MaxTokens: 100}}, "routing_view.max_tokens must be between 256 and 1048576"},
		"share":          {ModelDeployment{Provider: ModelRuntimeProvider, RoutingView: &DeploymentRoutingView{HeadShare: share(1.5)}}, "routing_view.head_share must be between 0 and 1"},
		"window":         {ModelDeployment{Provider: ModelRuntimeProvider, ScanWindow: &DeploymentScanWindow{Tokens: 64}}, "scan_window.tokens must be between 256 and 1048576"},
		"overlap":        {ModelDeployment{Provider: ModelRuntimeProvider, ScanWindow: &DeploymentScanWindow{Tokens: 1024, Overlap: overlap(512)}}, "scan_window.overlap must be at least 0 and less than half of scan_window.tokens (1024)"},
		"default window": {ModelDeployment{Provider: SystemOneProvider, ScanWindow: &DeploymentScanWindow{Overlap: overlap(3000)}}, "less than half of scan_window.tokens (4096)"},
		"max_tokens":     {ModelDeployment{Provider: ModelRuntimeProvider, ScanWindow: &DeploymentScanWindow{Tokens: 2048, MaxTokens: 8192}}, ""},
		"max_tokens below tokens": {
			ModelDeployment{Provider: ModelRuntimeProvider, ScanWindow: &DeploymentScanWindow{Tokens: 2048, MaxTokens: 1024}}, "scan_window.max_tokens must be at least scan_window.tokens (2048)",
		},
		"max_tokens below the default window": {
			ModelDeployment{Provider: ModelRuntimeProvider, ScanWindow: &DeploymentScanWindow{MaxTokens: 4096}}, "scan_window.max_tokens must be at least scan_window.tokens (8192)",
		},
		"reread_floor":      {ModelDeployment{Provider: ModelRuntimeProvider, ScanWindow: &DeploymentScanWindow{RereadFloor: share(1)}}, ""},
		"reread_floor zero": {ModelDeployment{Provider: ModelRuntimeProvider, ScanWindow: &DeploymentScanWindow{RereadFloor: share(0)}}, "scan_window.reread_floor must be greater than 0 and at most 1"},
		"reread_floor above one": {
			ModelDeployment{Provider: ModelRuntimeProvider, ScanWindow: &DeploymentScanWindow{RereadFloor: share(1.2)}}, "scan_window.reread_floor must be greater than 0 and at most 1",
		},
		"chat model": {ModelDeployment{Provider: OpenAIChatProvider, RoutingView: &DeploymentRoutingView{MaxTokens: 4096}}, "apply to model_runtime and systemone deployments"},
	} {
		err := tc.deployment.validateInputViews()
		if (tc.err == "") != (err == nil) || (err != nil && !strings.Contains(err.Error(), tc.err)) {
			t.Fatalf("%s: %v, want %q", name, err, tc.err)
		}
	}
}

func TestRoutingViewAndScanWindowParseFromTheCanonicalConfig(t *testing.T) {
	document := `version: v0.5
providers:
  defaults: {model: fast}
  models:
    - name: fast
      backend_refs: [{name: fast, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
global:
  model_catalog:
    deployments:
      hosted-d3:
        provider: systemone
        endpoint: https://decisions.example.com/v1/systemone
        model: d3
        auth: {api_key_env: ROUTING_VIEW_TEST_KEY}
        timeout_ms: 6000
        routing_view: {max_tokens: MAX, head_share: 0.3}
        scan_window: {tokens: 2048, overlap: 128}
`
	cfg, err := ParseYAMLBytes([]byte(strings.Replace(document, "MAX", "3000", 1)))
	if err != nil {
		t.Fatal(err)
	}
	deployment := cfg.ModelDeployments["hosted-d3"]
	if tokens, share := deployment.RoutingViewTokens(); tokens != 3000 || share != 0.3 {
		t.Fatalf("routing_view %d, %.2f", tokens, share)
	}
	if tokens, overlap := deployment.ScanWindowTokens(); tokens != 2048 || overlap != 128 {
		t.Fatalf("scan_window %d, %d", tokens, overlap)
	}
	if _, err := ParseYAMLBytes([]byte(strings.Replace(document, "MAX", "64", 1))); err == nil || !strings.Contains(err.Error(), "global.model_catalog.deployments.hosted-d3: routing_view.max_tokens must be between 256 and 1048576") {
		t.Fatalf("a view below the minimum loads: %v", err)
	}
}

func TestGuardRulesReadPartlyScannedInputBestEffortAndFailClosedWithoutAnAnswer(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.JailbreakRules = []JailbreakRule{{Name: "default"}, {Name: "strict", OnUnscanned: SystemOneUnscannedMatch, OnUnknown: string(RuleOnUnknownNoMatch)}}
	cfg.PIIRules = []PIIRule{{Name: "contact"}}
	cfg.SafetyRules = []SafetyRule{{Name: "harm"}}
	for _, want := range []struct {
		signalType, name, onUnscanned string
		failClosed                    bool
	}{
		{SignalTypeJailbreak, "default", SystemOneUnscannedBestEffort, true},
		{SignalTypeJailbreak, "strict", SystemOneUnscannedMatch, false},
		{SignalTypePII, "contact", SystemOneUnscannedBestEffort, true},
		{SignalTypeSafety, "harm", SystemOneUnscannedBestEffort, false},
	} {
		rule, ok := cfg.ApplicationRuleByName(want.signalType, want.name)
		if !ok || rule.Rule.EffectiveOnUnscanned() != want.onUnscanned || rule.FailClosed != want.failClosed || !rule.Rule.FullCoverage() {
			t.Fatalf("%s %s: on_unscanned %q, fails closed %v, coverage %q", want.signalType, want.name, rule.Rule.EffectiveOnUnscanned(), rule.FailClosed, rule.Rule.Coverage)
		}
	}
}
