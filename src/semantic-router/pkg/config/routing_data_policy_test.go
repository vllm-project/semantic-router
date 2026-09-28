package config

import "testing"

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
