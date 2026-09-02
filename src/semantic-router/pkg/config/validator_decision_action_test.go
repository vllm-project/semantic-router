package config

import (
	"strings"
	"testing"
)

func TestValidateDecisionAction(t *testing.T) {
	jailbreakRule := RuleNode{Type: SignalTypeJailbreak, Name: "prompt_injection"}
	tests := []struct {
		name    string
		rules   RuleNode
		action  *DecisionAction
		wantErr string
	}{
		{"no action", jailbreakRule, nil, ""},
		{"valid", jailbreakRule, &DecisionAction{Type: DecisionActionRoute, Destination: "safe-model"}, ""},
		{"invalid type", jailbreakRule, &DecisionAction{Type: "block", Destination: "safe-model"}, "action.type"},
		{"missing destination", jailbreakRule, &DecisionAction{Type: DecisionActionRoute}, "action.destination is required"},
		{"unknown destination", jailbreakRule, &DecisionAction{Type: DecisionActionRoute, Destination: "ghost"}, "not defined in model_config"},
		{"no jailbreak condition", RuleNode{Type: SignalTypeKeyword, Name: "x"}, &DecisionAction{Type: DecisionActionRoute, Destination: "safe-model"}, "jailbreak condition"},
		{"nested jailbreak condition", RuleNode{Operator: "AND", Conditions: []RuleNode{{Type: SignalTypeKeyword, Name: "x"}, jailbreakRule}}, &DecisionAction{Type: DecisionActionRoute, Destination: "safe-model"}, ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &RouterConfig{
				IntelligentRouting: IntelligentRouting{Decisions: []Decision{{Name: "guard", Rules: test.rules, Action: test.action}}},
				BackendModels:      BackendModels{ModelConfig: map[string]ModelParams{"safe-model": {}}},
			}
			err := validateDecisionModelContracts(cfg)
			if test.wantErr == "" && err != nil {
				t.Fatal(err)
			}
			if test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}
