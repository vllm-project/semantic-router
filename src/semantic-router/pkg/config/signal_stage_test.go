package config

import (
	"strings"
	"testing"
)

func TestSignalStageOfFollowsTheRuleNotTheType(t *testing.T) {
	cfg := stagedConfig()
	if got := cfg.SignalStageOf(SignalTypeJailbreak, "prompt_injection"); got != SignalStageRequest {
		t.Errorf("request-direction rule stage = %q", got)
	}
	if got := cfg.SignalStageOf(SignalTypeJailbreak, "unsafe_completion"); got != SignalStageResponse {
		t.Errorf("response-direction rule stage = %q", got)
	}
	if got := cfg.SignalStageOf("Jailbreak", "unsafe_completion"); got != SignalStageResponse {
		t.Errorf("type match must be case-insensitive like the engine, got %q", got)
	}
	if got := cfg.SignalStageOf(SignalTypeKeyword, "unsafe_completion"); got != SignalStageRequest {
		t.Errorf("a non-jailbreak type is request-stage, got %q", got)
	}
}

func TestRequestAndResponseJailbreakRulesSplitByDirection(t *testing.T) {
	cfg := stagedConfig()
	if got := cfg.RequestJailbreakRules(); len(got) != 1 || got[0].Name != "prompt_injection" {
		t.Errorf("RequestJailbreakRules = %+v", got)
	}
	if got := cfg.ResponseJailbreakRules(); len(got) != 1 || got[0].Name != "unsafe_completion" {
		t.Errorf("ResponseJailbreakRules = %+v", got)
	}
}

// A decision rule that names a response-direction rule is rejected wherever it
// sits in the tree: decisions are selected before the model has answered, so
// the rule could only ever read as unknown there.
func TestValidateRejectsDecisionReadingResponseDirectionRule(t *testing.T) {
	cfg := stagedConfig()
	cfg.Decisions = []Decision{
		{Name: "route", Rules: RuleCombination{Type: SignalTypeKeyword, Name: "probe"}},
		{Name: "guard", Rules: RuleCombination{
			Operator: "OR",
			Conditions: []RuleCondition{
				{Type: SignalTypeDomain, Name: "business"},
				{Operator: "AND", Conditions: []RuleCondition{
					{Type: SignalTypeKeyword, Name: "probe"},
					{Type: SignalTypeJailbreak, Name: "unsafe_completion"},
				}},
			},
		}},
	}
	err := validateSignalStageContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), `decision "guard"`) || !strings.Contains(err.Error(), `"unsafe_completion"`) {
		t.Fatalf("expected the nested response-direction reference to be rejected, got %v", err)
	}

	cfg.Decisions[1].Rules = RuleCombination{Type: SignalTypeJailbreak, Name: "prompt_injection"}
	if err := validateSignalStageContracts(cfg); err != nil {
		t.Fatalf("a request-direction rule is a decision input, got %v", err)
	}
}

// A projection is evaluated with the decisions, before the model has answered,
// and an input with no result takes its miss value. A response-direction rule
// behind a projection output a decision reads is therefore rejected at load,
// through a score, a score of scores, and a confidence-valued output.
func TestValidateRejectsProjectionReadingResponseDirectionRule(t *testing.T) {
	cfg := stagedConfig()
	cfg.Projections = Projections{
		Scores: []ProjectionScore{
			{Name: "completion_risk", Method: "weighted_sum", Inputs: []ProjectionScoreInput{
				{Type: SignalTypeJailbreak, Name: "unsafe_completion", Weight: 1},
			}},
			{Name: "combined_risk", Method: "weighted_sum", Inputs: []ProjectionScoreInput{
				{Type: SignalTypeProjection, Name: "completion_risk", Weight: 1},
			}},
			{Name: "confidence_risk", Method: "weighted_sum", Inputs: []ProjectionScoreInput{
				{Type: SignalTypeProjection, Name: "completion_flagged", ValueSource: ProjectionValueSourceConfidence, Weight: 1},
			}},
		},
		Mappings: []ProjectionMapping{
			{Name: "completion_band", Source: "completion_risk", Method: "threshold_bands", Outputs: []ProjectionMappingOutput{
				{Name: "completion_flagged", GTE: float64PtrForRoutingSignalUsageTest(0.5)},
			}},
			{Name: "combined_band", Source: "combined_risk", Method: "threshold_bands", Outputs: []ProjectionMappingOutput{
				{Name: "combined_flagged", GTE: float64PtrForRoutingSignalUsageTest(0.5)},
			}},
			{Name: "confidence_band", Source: "confidence_risk", Method: "threshold_bands", Outputs: []ProjectionMappingOutput{
				{Name: "confidence_flagged", GTE: float64PtrForRoutingSignalUsageTest(0.5)},
			}},
		},
	}
	outputs := []string{"completion_flagged", "combined_flagged", "confidence_flagged"}

	for _, output := range outputs {
		cfg.Decisions = []Decision{{Name: "guard", Rules: RuleCombination{
			Operator: "AND",
			Conditions: []RuleCondition{
				{Type: SignalTypeKeyword, Name: "probe"},
				{Type: SignalTypeProjection, Name: output},
			},
		}}}
		err := validateSignalStageContracts(cfg)
		if err == nil ||
			!strings.Contains(err.Error(), `decision "guard"`) ||
			!strings.Contains(err.Error(), `"unsafe_completion"`) ||
			!strings.Contains(err.Error(), `through projection "`+output+`"`) {
			t.Fatalf("output %q: expected the response-direction rule behind the projection to be rejected, got %v", output, err)
		}
	}

	cfg.Projections.Scores[0].Inputs[0].Name = "prompt_injection"
	for _, output := range outputs {
		cfg.Decisions = []Decision{{Name: "guard", Rules: RuleCombination{Type: SignalTypeProjection, Name: output}}}
		if err := validateSignalStageContracts(cfg); err != nil {
			t.Fatalf("output %q: a request-direction rule behind a projection is a decision input, got %v", output, err)
		}
	}
}

// A hallucination rule checks the model's answer, so it is response-stage by
// type: a decision reading one, directly or behind a projection, is rejected
// the way a response-direction jailbreak rule is.
func TestValidateRejectsDecisionReadingHallucinationRule(t *testing.T) {
	cfg := stagedConfig()
	cfg.HallucinationRules = []HallucinationRule{{Name: "ungrounded_claims"}}
	if got := cfg.SignalStageOf(SignalTypeHallucination, "ungrounded_claims"); got != SignalStageResponse {
		t.Fatalf("SignalStageOf(hallucination) = %q, want %q", got, SignalStageResponse)
	}

	cfg.Decisions = []Decision{{Name: "guard", Rules: RuleCombination{
		Operator: "AND",
		Conditions: []RuleCondition{
			{Type: SignalTypeKeyword, Name: "probe"},
			{Type: SignalTypeHallucination, Name: "ungrounded_claims"},
		},
	}}}
	err := validateSignalStageContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), `decision "guard"`) || !strings.Contains(err.Error(), `hallucination rule "ungrounded_claims"`) {
		t.Fatalf("expected the hallucination reference to be rejected, got %v", err)
	}

	cfg.Projections = Projections{
		Scores: []ProjectionScore{{Name: "grounding", Method: "weighted_sum", Inputs: []ProjectionScoreInput{
			{Type: SignalTypeHallucination, Name: "ungrounded_claims", Weight: 1},
		}}},
		Mappings: []ProjectionMapping{{Name: "grounding_band", Source: "grounding", Method: "threshold_bands", Outputs: []ProjectionMappingOutput{
			{Name: "ungrounded", GTE: float64PtrForRoutingSignalUsageTest(0.5)},
		}}},
	}
	cfg.Decisions = []Decision{{Name: "guard", Rules: RuleCombination{Type: SignalTypeProjection, Name: "ungrounded"}}}
	err = validateSignalStageContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), `through projection "ungrounded"`) {
		t.Fatalf("expected the hallucination rule behind the projection to be rejected, got %v", err)
	}
}

func TestJailbreakAndSafetyRulesDefaultToTheRequestStage(t *testing.T) {
	if got := (JailbreakRule{Name: "r"}).EffectiveStage(); got != SystemOneStageRequest {
		t.Fatalf("jailbreak stage = %q, want %q", got, SystemOneStageRequest)
	}
	if got := (JailbreakRule{Name: "r", Stage: SystemOneStageResponse}).EffectiveStage(); got != SystemOneStageResponse {
		t.Fatalf("jailbreak stage = %q, want %q", got, SystemOneStageResponse)
	}
	if got := (SafetyRule{Name: "r"}).EffectiveStage(); got != SystemOneStageRequest {
		t.Fatalf("safety stage = %q, want %q", got, SystemOneStageRequest)
	}
}

func stagedConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.JailbreakRules = []JailbreakRule{
		{Name: "prompt_injection"},
		{Name: "unsafe_completion", Stage: SystemOneStageResponse},
	}
	return cfg
}
