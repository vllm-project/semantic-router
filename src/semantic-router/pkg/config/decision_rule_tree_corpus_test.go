package config

import (
	"strings"
	"testing"
)

// ruleTreeCase is one entry of the decision rule-tree truth-table corpus: a rule
// tree plus the verdict every producer of the config contract must agree on.
// The nested-node reference contract is the CLI model (src/vllm-sr/cli/models.py,
// Condition.validate_node_shape); these cases pin the Go loader to it.
type ruleTreeCase struct {
	name    string
	rules   string
	wantErr string // "" means the tree must be accepted
}

// decisionRuleTreeConfig renders a minimal config whose single decision carries
// the given rules block.
func decisionRuleTreeConfig(rules string) []byte {
	return []byte(`
version: v0.3
listeners: []
providers:
  defaults:
    default_model: m1
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
          api_key: k
routing:
  modelCards:
    - name: m1
  signals:
    keywords:
      - name: k1
        operator: OR
        keywords:
          - alpha
  decisions:
    - name: d1
      priority: 1
      rules:
` + indentRuleBlock(rules) + `
      modelRefs:
        - model: m1
          use_reasoning: false
`)
}

func indentRuleBlock(rules string) string {
	lines := strings.Split(strings.TrimRight(rules, "\n"), "\n")
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		lines[i] = "        " + line
	}
	return strings.Join(lines, "\n")
}

func runRuleTreeCorpus(t *testing.T, corpus []ruleTreeCase) {
	t.Helper()
	for _, tc := range corpus {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(decisionRuleTreeConfig(tc.rules))
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("rule tree should be accepted, got: %v", err)
			case tc.wantErr != "" && err == nil:
				t.Fatalf("rule tree should be rejected with %q, got nil", tc.wantErr)
			case tc.wantErr != "" && !strings.Contains(err.Error(), tc.wantErr):
				t.Fatalf("error should mention %q, got: %v", tc.wantErr, err)
			}
		})
	}
}

// A combination operator is a closed set: the decision engine understands AND,
// OR and NOT and routes everything else to its default branch (OR), so an
// invalid operator widens the rule instead of failing.
var ruleTreeOperatorCorpus = []ruleTreeCase{
	{
		name:  "and_with_condition",
		rules: "operator: AND\nconditions:\n  - type: keyword\n    name: k1",
	},
	{
		name:  "or_lowercase",
		rules: "operator: or\nconditions:\n  - type: keyword\n    name: k1",
	},
	{
		name:  "root_not",
		rules: "operator: NOT\nconditions:\n  - type: keyword\n    name: k1",
	},
	{
		// An omitted operator stays valid, but its meaning is not settled across
		// producers: the engine evaluates it as OR while the CLI and the DSL
		// write AND. Configs should spell the operator out.
		name:  "omitted_operator",
		rules: "conditions:\n  - type: keyword\n    name: k1",
	},
	{
		name:    "unknown_operator",
		rules:   "operator: XOR\nconditions:\n  - type: keyword\n    name: k1",
		wantErr: `invalid rule operator "XOR" (valid: AND, OR, NOT)`,
	},
	{
		name:    "unknown_operator_lowercase",
		rules:   "operator: nor\nconditions:\n  - type: keyword\n    name: k1",
		wantErr: "invalid rule operator",
	},
	{
		// No evaluator trims the operator, so a padded value must not be blessed
		// here: it would reach the engine's default branch.
		name:    "operator_with_leading_space",
		rules:   "operator: \" NOT\"\nconditions:\n  - type: keyword\n    name: k1",
		wantErr: `invalid rule operator " NOT"`,
	},
	{
		name: "unknown_operator_nested",
		rules: "operator: AND\nconditions:\n  - operator: NONSENSE\n    conditions:\n" +
			"      - type: keyword\n        name: k1",
		wantErr: "rules.conditions[0]: invalid rule operator",
	},
}

func TestDecisionRuleTreeOperatorCorpus(t *testing.T) {
	runRuleTreeCorpus(t, ruleTreeOperatorCorpus)
}
