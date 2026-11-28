package config

import (
	"strings"
	"testing"
)

func TestValidateDecisionOnUnknown(t *testing.T) {
	tests := []struct {
		name    string
		rules   RuleNode
		wantErr string
	}{
		{"valid", RuleNode{Operator: "AND", OnUnknown: RuleOnUnknownFailRequest}, ""},
		{"invalid value", RuleNode{Operator: "AND", OnUnknown: "allow"}, "on_unknown"},
		{"nested", RuleNode{Operator: "AND", Conditions: []RuleNode{{Type: SignalTypeKeyword, Name: "x", OnUnknown: RuleOnUnknownMatch}}}, "root rules node"},
		{"conflicts with on_error", RuleNode{Operator: "AND", OnUnknown: RuleOnUnknownNoMatch, Conditions: []RuleNode{{Type: SignalTypeKeyword, Name: "x", OnError: "no_match"}}}, "on_error has no effect"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &RouterConfig{IntelligentRouting: IntelligentRouting{Decisions: []Decision{{Name: "route", Rules: test.rules}}}}
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

func TestParseYAMLBytesRejectsOnUnknownOnErrorConflict(t *testing.T) {
	yaml := buildDecisionRefConfig(`      - name: guarded
        priority: 1
        rules:
          operator: AND
          on_unknown: no_match
          conditions:
            - {type: systemone, name: risk, label: RISKY, on_error: no_match}
        modelRefs:
          - model: m1
    signals:
      systemone:
        - name: risk
          question: {type: choice, instructions: "Is the request risky?", choices: [{key: SAFE}, {key: RISKY}]}
`)
	_, err := ParseYAMLBytes(yaml)
	if err == nil || !strings.Contains(err.Error(), OnUnknownOnErrorConflictMessage) {
		t.Fatalf("error = %v, want the on_unknown + on_error conflict rejection", err)
	}
}

func TestUnguardedConditions(t *testing.T) {
	rules := RuleNode{Operator: "AND", Conditions: []RuleNode{
		{Type: SignalTypeSystemOne, Name: "risk", Label: "RISKY"},
		{Type: SignalTypeSystemOne, Name: "safe", Label: "SAFE", OnError: "match"},
		{Operator: "OR", Conditions: []RuleNode{{Type: SignalTypeSystemOne, Name: "nested"}}},
		{Type: SignalTypeKeyword, Name: "kw"},
	}}
	if got := unguardedConditionNames(&RouterConfig{}, &rules); strings.Join(got, ",") != "risk,nested" {
		t.Fatalf("unguarded = %v", got)
	}
}

func TestUnguardedRemoteConditions(t *testing.T) {
	rules := RuleNode{Operator: "OR", Conditions: []RuleNode{
		{Type: SignalTypeDomain, Name: "math"},
		{Type: SignalTypeComplexity, Name: "hard"},
		{Type: SignalTypePII, Name: "pii"},
		{Type: SignalTypeJailbreak, Name: "guard"},
		{Type: SignalTypeSafety, Name: "unsafe"},
		{Type: SignalTypeSafety, Name: "local"},
		{Type: SignalTypeSafety, Name: "inline"},
	}}
	cfg := &RouterConfig{}
	cfg.Categories = []Category{{Name: "math"}}
	cfg.ComplexityRules = []ComplexityRule{{Name: "hard"}}
	cfg.PIIRules = []PIIRule{{Name: "pii"}}
	cfg.JailbreakRules = []JailbreakRule{{Name: "guard"}}
	cfg.SafetyRules = []SafetyRule{{Name: "unsafe"}, {Name: "local", Deployment: "local"}, {Name: "inline", Deployment: "remote"}}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"remote": {Provider: SystemOneProvider},
		"local":  {Provider: ModelRuntimeProvider},
	}
	cfg.GlobalModelBindings = map[string]ModelBinding{
		SignalTypeDomain:     {Deployment: "remote"},
		SignalTypeComplexity: {Deployment: "remote"},
		SignalTypePII:        {Deployment: "remote"},
		SignalTypeJailbreak:  {Deployment: "remote"},
		SignalTypeSafety:     {Deployment: "remote"},
	}
	if got := unguardedConditionNames(cfg, &rules); strings.Join(got, ",") != "math,hard,unsafe,inline" {
		t.Fatalf("unguarded = %v", got)
	}
	cfg.GlobalModelBindings = map[string]ModelBinding{SignalTypeSafety: {Deployment: "local"}}
	cfg.ModelBindings = map[string]ModelBinding{SignalTypeDomain: {Deployment: "remote"}}
	if got := unguardedConditionNames(cfg, &rules); strings.Join(got, ",") != "math,inline" {
		t.Fatalf("unguarded with a recipe binding = %v", got)
	}
	cfg.ModelBindings = nil
	cfg.GlobalModelBindings = nil
	if got := unguardedConditionNames(cfg, &rules); strings.Join(got, ",") != "inline" {
		t.Fatalf("unguarded without remote bindings = %v", got)
	}
}

func unguardedConditionNames(cfg *RouterConfig, rules *RuleNode) []string {
	var names []string
	for _, node := range unguardedConditions(cfg, rules) {
		names = append(names, node.Name)
	}
	return names
}
