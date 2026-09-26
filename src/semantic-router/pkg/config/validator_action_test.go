package config

import (
	"fmt"
	"strings"
	"testing"
)

func actionRoutingYAML(declared []string, referenced string) string {
	var signals strings.Builder
	for _, name := range declared {
		fmt.Fprintf(&signals, "      - name: %s\n", name)
	}
	return fmt.Sprintf(`version: v0.3
providers:
  defaults:
    model: code-model
  models:
    - name: code-model
      backend_refs:
        - name: primary
          endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: code-model
  signals:
    actions:
%s  decisions:
    - name: action-route
      priority: 100
      rules:
        type: action
        name: %s
      modelRefs:
        - model: code-model
`, signals.String(), referenced)
}

func TestActionRulesLoadAndExportThroughCanonicalConfig(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(actionRoutingYAML(SupportedActions(), ActionFix)))
	if err != nil {
		t.Fatalf("canonical config with every action failed to load: %v", err)
	}
	loaded := collectActionRuleNames(cfg.ActionRules)
	for _, action := range SupportedActions() {
		if _, ok := loaded[action]; !ok {
			t.Fatalf("loaded action rules = %v, missing %q", cfg.ActionRules, action)
		}
	}
	if exported := canonicalSignalsFromSignals(cfg.Signals).Actions; len(exported) != len(SupportedActions()) {
		t.Fatalf("exported actions = %v, want all %d", exported, len(SupportedActions()))
	}
}

func TestActionRulesRejectNamesOutsideTheVocabulary(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(actionRoutingYAML([]string{"review"}, "review")))
	want := `routing.signals.actions[0].name "review" must be one of generate, explain, fix, refactor, test, other`
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("expected vocabulary error %q, got %v", want, err)
	}
}

func TestActionDecisionMustReferenceADeclaredAction(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(actionRoutingYAML([]string{ActionExplain}, ActionFix)))
	if err == nil || !strings.Contains(err.Error(), `signal action("fix") is not declared`) {
		t.Fatalf("expected an undeclared action reference error, got %v", err)
	}
}
