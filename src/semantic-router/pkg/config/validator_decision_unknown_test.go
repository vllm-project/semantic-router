package config

import (
	"strings"
	"testing"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
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

func TestUnguardedSystemOneConditions(t *testing.T) {
	rules := RuleNode{Operator: "AND", Conditions: []RuleNode{
		{Type: SignalTypeSystemOne, Name: "risk", Label: "RISKY"},
		{Type: SignalTypeSystemOne, Name: "safe", Label: "SAFE", OnError: "match"},
		{Operator: "OR", Conditions: []RuleNode{{Type: SignalTypeSystemOne, Name: "nested"}}},
		{Type: SignalTypeKeyword, Name: "kw"},
	}}
	got := unguardedSystemOneConditions(&rules)
	if len(got) != 2 || got[0] != "risk" || got[1] != "nested" {
		t.Fatalf("unguarded = %v", got)
	}
}

// A condition on an image rule is also unknown when the request carries an
// image no decision model reads, so its startup warning says so.
func TestUnguardedImageConditionWarningNamesUnreadableImages(t *testing.T) {
	core, logs := observer.New(zapcore.WarnLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	yaml := buildDecisionRefConfig(`      - name: guarded
        priority: 1
        rules:
          operator: AND
          conditions:
            - {type: systemone, name: unsafe_image}
            - {type: systemone, name: risk, label: RISKY}
        modelRefs:
          - model: m1
    signals:
      systemone:
        - name: unsafe_image
          deployment: vision
          input: {image: true}
          question: {type: noul, instructions: "Does the image show prohibited content?"}
        - name: risk
          question: {type: choice, instructions: "Is the request risky?", choices: [{key: SAFE}, {key: RISKY}]}
global:
  model_catalog:
    deployments:
      vision:
        provider: systemone
        endpoint: http://127.0.0.1:9/v1/systemone
        model: vision-guard
        capabilities: {question_types: [noul], modalities: [text, image]}
`)
	if _, err := ParseYAMLBytes(yaml); err != nil {
		t.Fatal(err)
	}
	warned := map[string]string{}
	for _, entry := range logs.All() {
		for _, name := range []string{"unsafe_image", "risk"} {
			if strings.Contains(entry.Message, `condition "`+name+`"`) {
				warned[name] = entry.Message
			}
		}
	}
	if !strings.Contains(warned["unsafe_image"], "an image") || !strings.Contains(warned["unsafe_image"], "on_unknown: match") {
		t.Fatalf("image condition warning = %q", warned["unsafe_image"])
	}
	if warned["risk"] == "" || strings.Contains(warned["risk"], "an image") {
		t.Fatalf("text condition warning = %q", warned["risk"])
	}
}
