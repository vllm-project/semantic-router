package config

import (
	"strings"
	"testing"
)

func stagedConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.JailbreakRules = []JailbreakRule{
		{Name: "prompt_injection", Threshold: 0.8},
		{Name: "unsafe_completion", Threshold: 0.85, Direction: SignalDirectionResponse},
	}
	return cfg
}

func TestJailbreakRuleStageDefaultsToRequest(t *testing.T) {
	if got := (JailbreakRule{Name: "r"}).Stage(); got != SignalStageRequest {
		t.Fatalf("empty direction stage = %q, want %q", got, SignalStageRequest)
	}
	if got := (JailbreakRule{Name: "r", Direction: SignalDirectionResponse}).Stage(); got != SignalStageResponse {
		t.Fatalf("direction response stage = %q, want %q", got, SignalStageResponse)
	}
}

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

// A decision is response-stage wherever the response-direction rule sits in
// its tree: composed with a request signal under OR it would otherwise be
// selectable at request time off the request half alone.
func TestDecisionStageReadsNestedConditions(t *testing.T) {
	cfg := stagedConfig()
	nested := &RuleNode{
		Operator: "OR",
		Conditions: []RuleCondition{
			{Type: SignalTypeDomain, Name: "business"},
			{Operator: "AND", Conditions: []RuleCondition{
				{Type: SignalTypeKeyword, Name: "probe"},
				{Type: SignalTypeJailbreak, Name: "unsafe_completion"},
			}},
		},
	}
	if got := cfg.DecisionStage(nested); got != SignalStageResponse {
		t.Errorf("nested response-direction rule: stage = %q", got)
	}
	requestOnly := &RuleNode{Type: SignalTypeJailbreak, Name: "prompt_injection"}
	if got := cfg.DecisionStage(requestOnly); got != SignalStageRequest {
		t.Errorf("request-direction rule: stage = %q", got)
	}
	if got := cfg.DecisionStage(nil); got != SignalStageRequest {
		t.Errorf("nil rules: stage = %q", got)
	}
}

func TestDecisionsAtStage(t *testing.T) {
	cfg := stagedConfig()
	decisions := []Decision{
		{Name: "route", Rules: RuleCombination{Type: SignalTypeKeyword, Name: "probe"}},
		{Name: "guard", Rules: RuleCombination{Type: SignalTypeJailbreak, Name: "unsafe_completion"}},
	}
	if got := cfg.DecisionsAtStage(decisions, SignalStageRequest); len(got) != 1 || got[0].Name != "route" {
		t.Errorf("request-stage decisions = %+v", got)
	}
	if got := cfg.DecisionsAtStage(decisions, SignalStageResponse); len(got) != 1 || got[0].Name != "guard" {
		t.Errorf("response-stage decisions = %+v", got)
	}
}

func TestValidateDecisionStagesRejectsResponseOnlyConfig(t *testing.T) {
	cfg := stagedConfig()
	cfg.Decisions = []Decision{{
		Name:  "guard",
		Rules: RuleCombination{Type: SignalTypeJailbreak, Name: "unsafe_completion"},
	}}
	err := validateSignalStageContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), "request time") {
		t.Fatalf("expected the response-only configuration to be rejected, got %v", err)
	}

	cfg.Decisions = append(cfg.Decisions, Decision{
		Name:  "route",
		Rules: RuleCombination{Type: SignalTypeKeyword, Name: "probe"},
	})
	if err := validateSignalStageContracts(cfg); err != nil {
		t.Fatalf("one request-stage decision is enough, got %v", err)
	}
}

func TestValidateJailbreakContractsDirection(t *testing.T) {
	cases := []struct {
		name string
		rule JailbreakRule
		want string
	}{
		{
			name: "unknown direction",
			rule: JailbreakRule{Name: "r", Threshold: 0.5, Direction: "Response"},
			want: "unknown direction",
		},
		{
			name: "contrastive cannot score a response",
			rule: JailbreakRule{Name: "r", Threshold: 0.5, Direction: SignalDirectionResponse, Method: JailbreakMethodContrastive, JailbreakPatterns: []string{"x"}},
			want: "request-stage only",
		},
		{
			name: "include_history has no meaning on a response",
			rule: JailbreakRule{Name: "r", Threshold: 0.5, Direction: SignalDirectionResponse, IncludeHistory: true},
			want: "include_history",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.JailbreakRules = []JailbreakRule{tc.rule}
			err := validateJailbreakContracts(cfg)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want it to mention %q", err, tc.want)
			}
		})
	}

	cfg := &RouterConfig{}
	cfg.JailbreakRules = []JailbreakRule{
		{Name: "req", Threshold: 0.5, Direction: SignalDirectionRequest},
		{Name: "resp", Threshold: 0.5, Direction: SignalDirectionResponse, Method: JailbreakMethodClassifier},
	}
	if err := validateJailbreakContracts(cfg); err != nil {
		t.Fatalf("valid directions rejected: %v", err)
	}
}
