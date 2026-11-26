package config

import (
	"strings"
	"testing"
)

// A request that matches no decision runs the guard of global.plugins, so its
// rules name signals of the recipe even when every decision replaces them.
func TestTheDefaultGuardNamesSignalsOfTheRecipe(t *testing.T) {
	doc := `version: v0.5
providers:
  models:
    - name: model-a
      backend_refs: [{endpoint: "127.0.0.1:8000", provider: vllm}]
recipes:
  - name: default
    signals:
      keywords: [{name: declared, operator: OR, keywords: [secret]}]
    decisions:
      - name: open
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: model-a}]
        plugins:
          - type: response_guard
            configuration: {rules: [{when: {type: keyword, name: declared}, action: none}]}
global:
  plugins:
    - type: response_guard
      configuration: {rules: [{when: {type: keyword, name: missing}, action: block}]}
`
	_, err := ParseYAMLBytes([]byte(doc))
	if err == nil || !strings.Contains(err.Error(), `"recipes[default].plugins" response_guard rules[0]: signal keyword("missing") is not declared in this recipe`) {
		t.Fatalf("err = %v, want the default guard's undeclared signal refused", err)
	}
}

// An annotate-only guard opts into holding a streamed answer with hold_stream,
// without changing the default or an unrelated guard's behavior.
func TestResponseGuardHoldStreamOptIn(t *testing.T) {
	jailbreak := RuleNode{Type: SignalTypeJailbreak, Name: "jailbreak"}
	annotate := ResponseGuardPluginConfig{
		Rules: []ResponseGuardRule{{When: jailbreak, Action: ResponseGuardAnnotate}},
	}
	if annotate.Enforces() {
		t.Fatal("an annotate-only guard holds a stream by default")
	}
	annotate.HoldStream = true
	if !annotate.Enforces() {
		t.Fatal("hold_stream did not make an annotate-only guard enforce")
	}
	// A guard that already holds needs no opt-in, and the opt-in leaves other
	// rules' actions unchanged.
	blocking := ResponseGuardPluginConfig{
		Rules:      []ResponseGuardRule{{When: jailbreak, Action: ResponseGuardBlock}},
		HoldStream: true,
	}
	if !blocking.Enforces() {
		t.Fatal("a blocking guard must hold a stream")
	}
	none := ResponseGuardPluginConfig{
		Rules:      []ResponseGuardRule{{When: jailbreak, Action: ResponseGuardNone}},
		HoldStream: true,
	}
	if !none.Enforces() {
		t.Fatal("hold_stream must hold even when the rule only records")
	}
}
