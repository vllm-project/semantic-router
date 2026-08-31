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
