package config

import (
	"strings"
	"testing"
)

func TestRoutingReplayPolicyOnlyTightensExistingEnablement(t *testing.T) {
	yes, no := true, false
	for _, policy := range []*RoutingDataPolicy{nil, {}, {Replay: &yes}, {Replay: &no}} {
		for _, global := range []bool{false, true} {
			for _, enabled := range []*bool{nil, &yes, &no} {
				cfg := &RouterConfig{RouterReplay: RouterReplayConfig{Enabled: global}}
				cfg.DataPolicy = policy
				var decision *Decision
				wantEnabled := global
				if enabled != nil {
					payload, err := NewStructuredPayload(RouterReplayPluginConfig{Enabled: *enabled})
					if err != nil {
						t.Fatal(err)
					}
					decision = &Decision{Name: "ordinary", Plugins: []DecisionPlugin{{Type: DecisionPluginRouterReplay, Configuration: payload}}}
					wantEnabled = *enabled
				}
				if policy != nil && policy.Replay != nil && !*policy.Replay {
					wantEnabled = false
				}
				got := cfg.EffectiveRouterReplayConfig(decision)
				if (got != nil && got.Enabled) != wantEnabled {
					t.Fatalf("policy=%+v global=%v decision=%v: got=%+v, want enabled=%v", policy, global, enabled, got, wantEnabled)
				}
			}
		}
	}
}

func TestRoutingDataPolicyClonePreservesOptionalValues(t *testing.T) {
	var absent *RoutingDataPolicy
	if absent.Clone() != nil || !absent.ReplayAllowed() {
		t.Fatal("absent policy must add no restriction")
	}
	empty := (&RoutingDataPolicy{}).Clone()
	if empty == nil || empty.Replay != nil || !empty.ReplayAllowed() {
		t.Fatal("empty policy lost its unset replay value")
	}
	for _, enabled := range []bool{false, true} {
		original := &RoutingDataPolicy{Replay: &enabled}
		clone := original.Clone()
		if clone == original || clone.Replay == original.Replay || *clone.Replay != enabled {
			t.Fatal("clone did not preserve and separate the replay preference")
		}
		*clone.Replay = !enabled
		if *original.Replay != enabled {
			t.Fatal("mutating clone changed the source policy")
		}
	}
}

func personalDataPolicyYAML(policy string, pii bool) []byte {
	signals := ""
	if pii {
		signals = "  signals:\n    pii:\n      - name: personal_data\n        pii_types_allowed: []\n"
	}
	return []byte(`version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
routing:
` + policy + signals + `  decisions:
    - name: everything
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: general}]
`)
}

func TestAPersonalDataReplayLimitAsksThePIISignals(t *testing.T) {
	cfg, err := ParseYAMLBytes(personalDataPolicyYAML("  data_policy:\n    replay_personal_data: false\n", true))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.DataPolicy.ReplayPersonalDataAllowed() {
		t.Fatal("replay_personal_data: false must forbid replaying personal data")
	}
	if !cfg.UsesSignalTypeInReachableRouting(SignalTypePII) {
		t.Fatal("the PII signal must be asked when only the data policy reads it")
	}
	unrestricted, err := ParseYAMLBytes(personalDataPolicyYAML("", true))
	if err != nil {
		t.Fatal(err)
	}
	if unrestricted.UsesSignalTypeInReachableRouting(SignalTypePII) {
		t.Fatal("an unreferenced PII signal stays unasked without the data policy")
	}
}

func TestAPersonalDataReplayLimitNeedsAPIISignal(t *testing.T) {
	_, err := ParseYAMLBytes(personalDataPolicyYAML("  data_policy:\n    replay_personal_data: false\n", false))
	if err == nil || !strings.Contains(err.Error(), "replay_personal_data: false needs a routing.signals.pii rule") {
		t.Fatalf("want the missing PII rule error, got %v", err)
	}
}

func TestRoutingDataPolicyCloneKeepsThePersonalDataLimit(t *testing.T) {
	no := false
	policy := &RoutingDataPolicy{ReplayPersonalData: &no}
	cloned := policy.Clone()
	no = true
	if cloned.ReplayPersonalDataAllowed() {
		t.Fatal("the clone must keep its own false value")
	}
}
