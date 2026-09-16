package extproc

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestSessionProtectionRecordsEligibilitySeparatelyFromSparseScores(t *testing.T) {
	ctx := &selection.SelectionContext{
		CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "proposal"}},
		AgenticSession:  &selection.AgenticSessionContext{PreviousModel: "current"},
	}
	base := &selection.SelectionResult{
		SelectedModel: "proposal", Method: selection.MethodStatic,
		AllScores: map[string]float64{"proposal": 0},
	}
	identity := routerLearningIdentity{scope: config.RouterLearningScopeSession}
	result, held := sessionScopeProtectedResult(config.RouterLearningProtectionConfig{}, base, ctx, identity)
	if !held || result.SelectedModel != "current" {
		t.Fatal("session protection must retain the eligible current model")
	}
	policy := newRouterLearningPolicy(routerLearningMethodProtection)
	policy.Details.Protection = newRouterLearningProtectionDiagnostics(result.SessionPolicy, routerLearningIdentityDiagnostics{})
	diagnostics := policy.toReplayProtection()
	if !reflect.DeepEqual(diagnostics.CandidateModels, []string{"current", "proposal"}) ||
		!reflect.DeepEqual(diagnostics.BaseScores, map[string]float64{"proposal": 0}) {
		t.Fatalf("eligibility must not be inferred from scores or invent missing scores: %+v", diagnostics)
	}
	raw, err := json.Marshal(diagnostics)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Candidates []string `json:"candidate_models"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil || len(wire.Candidates) != 2 {
		t.Fatalf("candidate inventory lost in replay JSON: %s (%v)", raw, err)
	}
	result.SessionPolicy.CandidateModels[0] = "changed"
	if diagnostics.CandidateModels[0] != "current" {
		t.Fatal("replay candidate inventory must own its snapshot")
	}
	ctx.CandidateModels = ctx.CandidateModels[1:]
	if _, held := sessionScopeProtectedResult(config.RouterLearningProtectionConfig{}, base, ctx, identity); held {
		t.Fatal("recorded candidates must not authorize a model outside the current pool")
	}
}

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
