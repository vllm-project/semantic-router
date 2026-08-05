package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestRouterLearningPolicySerializationKeepsCommonFieldsAuthoritative(t *testing.T) {
	policy := newRouterLearningPolicy(routerLearningMethodProtection)
	policy.Mode = config.DecisionAdaptationModeApply
	policy.Scope = config.RouterLearningScopeConversation
	policy.Action = routerLearningActionHoldCurrent
	policy.Reason = "tool_or_protocol_state"
	policy.Details.Protection = newRouterLearningProtectionDiagnostics(
		&selection.SessionPolicyTrace{
			Phase:          "provider_state",
			CurrentModel:   "qwen-small",
			SelectedModel:  "qwen-small",
			HardLocked:     true,
			HardLockReason: "tool_loop",
		},
		routerLearningIdentityDiagnostics{},
	)

	serialized := policy.ToMap()
	if got, _ := serialized["learning"].(string); got != routerLearningPolicyName {
		t.Fatalf("expected learning marker, got %#v", serialized)
	}
	if got, _ := serialized["method"].(string); got != string(routerLearningMethodProtection) {
		t.Fatalf("expected common method field to be authoritative, got %#v", serialized)
	}
	if got, _ := serialized["action"].(string); got != string(routerLearningActionHoldCurrent) {
		t.Fatalf("expected common action field to be authoritative, got %#v", serialized)
	}
	if got := policy.SessionPhase(); got != "provider_state" {
		t.Fatalf("expected typed session phase accessor, got %q", got)
	}
	if !policy.HardLocked() {
		t.Fatalf("expected typed hard lock accessor")
	}
}

func TestRouterLearningPoliciesFilterEmptyPolicies(t *testing.T) {
	policies := routerLearningPolicies{}
	policies.Set(newRouterLearningPolicy(routerLearningMethodProtection))
	policies.Set(routerLearningPolicy{})
	policies.Set(routerLearningPolicy{
		Method: routerLearningMethodProtection,
		Action: routerLearningActionAllowSwitch,
	})

	if _, ok := policies.Policy(routerLearningMethodAdaptation); ok {
		t.Fatalf("expected no empty adaptation policy, got %#v", policies)
	}
	got, ok := policies.Policy(routerLearningMethodProtection)
	if !ok || got.Action != routerLearningActionAllowSwitch {
		t.Fatalf("expected non-empty protection policy, got %#v", policies)
	}
}

func TestRouterLearningAdaptationStrategyRegistryResolvesDefaultStrategy(t *testing.T) {
	strategy, ok := routerLearningAdaptationStrategies.Strategy(config.RouterLearningAdaptationConfig{})

	if !ok || strategy == nil || strategy.Name() != config.RouterLearningStrategyRoutingSampling {
		t.Fatalf("expected default routing_sampling strategy, got strategy=%#v ok=%v", strategy, ok)
	}
}

func TestRouterLearningAdaptationStrategyRegistryRejectsUnknownStrategy(t *testing.T) {
	strategy, ok := routerLearningAdaptationStrategies.Strategy(config.RouterLearningAdaptationConfig{
		Strategy: "missing_strategy",
	})

	if ok || strategy != nil {
		t.Fatalf("expected unknown strategy to be unavailable, got strategy=%#v ok=%v", strategy, ok)
	}
}

func TestProtectionReplayDiagnosticsNormalizesInternalAlgorithm(t *testing.T) {
	policy := newRouterLearningPolicy(routerLearningMethodProtection)
	policy.Mode = config.DecisionAdaptationModeApply
	policy.Scope = config.RouterLearningScopeConversation
	policy.Action = routerLearningActionAllowSwitch
	policy.Reason = "switch_allowed"
	policy.Details.Protection = newRouterLearningProtectionDiagnostics(
		&selection.SessionPolicyTrace{
			Algorithm:     "agentic_continuity_routing",
			BaseMethod:    "hybrid",
			SelectedModel: "model-b",
		},
		routerLearningIdentityDiagnostics{},
	)

	diagnostics := policy.toReplayProtection()
	if diagnostics == nil {
		t.Fatal("expected replay protection diagnostics")
	}
	if diagnostics.Method != string(routerLearningMethodProtection) ||
		diagnostics.Algorithm != string(routerLearningMethodProtection) ||
		diagnostics.BaseMethod != "hybrid" {
		t.Fatalf("expected protection-facing replay diagnostics, got %#v", diagnostics)
	}
}

func testLearningPolicies(policies ...routerLearningPolicy) routerLearningPolicies {
	out := routerLearningPolicies{}
	for _, policy := range policies {
		out.Set(policy)
	}
	return out
}
