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

func TestUnguardedConditions(t *testing.T) {
	rules := RuleNode{Operator: "AND", Conditions: []RuleNode{
		{Type: SignalTypeClassifier, Name: "risk", Label: "RISKY"},
		{Type: SignalTypeClassifier, Name: "safe", Label: "SAFE", OnError: "match"},
		{Operator: "OR", Conditions: []RuleNode{{Type: SignalTypeClassifier, Name: "nested"}}},
		{Type: SignalTypeKeyword, Name: "kw"},
	}}
	if got := unguardedConditionNames(&RouterConfig{}, &rules); strings.Join(got, ",") != "risk,nested" {
		t.Fatalf("unguarded = %v", got)
	}
}

func TestUnguardedRemoteConditions(t *testing.T) {
	rules := RuleNode{Operator: "OR", Conditions: []RuleNode{
		{Type: SignalTypeDomain, Name: "math"},
		{Type: SignalTypeComplexity, Name: "reasoning:hard"},
		{Type: SignalTypePII, Name: "pii"},
		{Type: SignalTypeJailbreak, Name: "guard"},
		{Type: SignalTypeSafety, Name: "unsafe"},
		{Type: SignalTypeSafety, Name: "local"},
		{Type: SignalTypeSafety, Name: "hazard"},
		{Type: SignalTypeSafety, Name: "inline"},
	}}
	cfg := &RouterConfig{}
	cfg.SafetyRules = []SafetyRule{{Name: "local"}, {Name: "inline", Hazard: &SafetyHazardRule{Model: "remote-hazard"}}}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"remote": {Provider: "http"},
		"local":  {Provider: ModelRuntimeProvider},
	}
	cfg.ModelBindings = map[string]ModelBinding{
		"domain_classifier":    {Deployment: "remote"},
		"complexity":           {Deployment: "remote"},
		"pii_classifier":       {Deployment: "remote"},
		"prompt_guard":         {Deployment: "remote"},
		"safety.unsafe":        {Deployment: "remote"},
		"safety.local":         {Deployment: "local"},
		"safety.hazard.hazard": {Deployment: "remote"},
	}
	if got := unguardedConditionNames(cfg, &rules); strings.Join(got, ",") != "math,reasoning:hard,pii,guard,unsafe,hazard,inline" {
		t.Fatalf("unguarded = %v", got)
	}
	cfg.PIIModel.OnError = OnErrorBlock
	cfg.PromptGuard.OnError = OnErrorBlock
	if got := unguardedConditionNames(cfg, &rules); strings.Join(got, ",") != "math,reasoning:hard,unsafe,hazard,inline" {
		t.Fatalf("unguarded with block on_error = %v", got)
	}
	cfg.ModelBindings = nil
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
