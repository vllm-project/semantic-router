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
