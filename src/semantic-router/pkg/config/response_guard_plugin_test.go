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
