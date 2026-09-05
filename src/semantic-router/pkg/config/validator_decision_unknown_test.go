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
	canonicalYAML := []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    default_model: qwen2.5:3b
  models:
    - name: qwen2.5:3b
      provider_model_id: served-qwen
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:11434
          protocol: http
routing:
  modelCards:
    - name: qwen2.5:3b
      param_size: 3b
  signals:
    classifiers:
      - name: risk
        type: local
        model_path: models/risk
        labels: [SAFE, RISKY]
        use_cpu: true
  decisions:
    - name: guarded
      priority: 100
      rules:
        operator: AND
        on_unknown: no_match
        conditions:
          - type: classifier
            name: risk
            label: RISKY
            predicate:
              gte: 0.5
            on_error: no_match
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
`)
	_, err := ParseYAMLBytes(canonicalYAML)
	if err == nil || !strings.Contains(err.Error(), "on_error has no effect") {
		t.Fatalf("error = %v, want the on_unknown + on_error conflict rejection", err)
	}
}
