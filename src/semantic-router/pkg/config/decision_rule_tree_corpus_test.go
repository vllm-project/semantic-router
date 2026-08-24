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

// NOT is strictly unary. evalNOT logs a warning and reports a non-match for any
// other child count, so a mis-shaped NOT turns its decision into one that can
// never match instead of failing at load. (The IntelligentRoute CRD comment
// promising NOR semantics for multiple children was never implemented.)
var ruleTreeArityCorpus = []ruleTreeCase{
	{
		name:    "not_without_children",
		rules:   "operator: NOT\nconditions: []",
		wantErr: "NOT requires exactly one child condition, got 0",
	},
	{
		name: "not_with_two_children",
		rules: "operator: NOT\nconditions:\n  - type: keyword\n    name: k1\n" +
			"  - type: keyword\n    name: k1",
		wantErr: "NOT requires exactly one child condition, got 2",
	},
	{
		name: "nested_not_with_two_children",
		rules: "operator: AND\nconditions:\n  - operator: not\n    conditions:\n" +
			"      - type: keyword\n        name: k1\n      - type: keyword\n        name: k1",
		wantErr: "rules.conditions[0]: NOT requires exactly one child condition",
	},
}

func TestDecisionRuleTreeArityCorpus(t *testing.T) {
	runRuleTreeCorpus(t, ruleTreeArityCorpus)
}

// A node is either a leaf (a signal reference) or a combination of children.
// IsLeaf keys off `type`, so a node that mixes the shapes, or carries leaf
// fields without a type, silently loses half of what the author wrote — and a
// combination with no children is dead (OR) or unconditional (AND) rather than
// the rule that was written.
var ruleTreeShapeCorpus = []ruleTreeCase{
	{
		name:  "root_bare_leaf",
		rules: "type: keyword\nname: k1",
	},
	{
		name:  "root_without_rules",
		rules: "{}",
	},
	{
		name:  "root_match_all",
		rules: "operator: AND\nconditions: []",
	},
	{
		name: "leaf_and_combination_mixed",
		rules: "type: keyword\nname: k1\noperator: OR\nconditions:\n" +
			"  - type: keyword\n    name: k1",
		wantErr: "must be either a leaf (type/name) or a combination (operator/conditions), not both",
	},
	{
		name:    "combination_with_leaf_label",
		rules:   "label: high\noperator: AND\nconditions:\n  - type: keyword\n    name: k1",
		wantErr: "not both",
	},
	{
		// Evaluates as an empty OR: a decision that can never match.
		name:    "root_name_without_type",
		rules:   "name: k1",
		wantErr: "rules: leaf condition requires a type",
	},
	{
		// Also satisfies IsEmpty, so the engine short-circuits it into an
		// unconditional match.
		name:    "root_label_without_type",
		rules:   "label: high",
		wantErr: "rules: leaf condition requires a type",
	},
	{
		name:    "nested_leaf_without_type",
		rules:   "operator: AND\nconditions:\n  - name: k1",
		wantErr: "rules.conditions[0]: leaf condition requires a type",
	},
	{
		name:    "leaf_with_children",
		rules:   "type: keyword\nname: k1\nconditions:\n  - type: keyword\n    name: k1",
		wantErr: "leaf condition cannot declare child conditions",
	},
	{
		name: "nested_leaf_with_children",
		rules: "operator: AND\nconditions:\n  - type: keyword\n    name: k1\n" +
			"    conditions:\n      - type: keyword\n        name: k1",
		wantErr: "rules.conditions[0]: leaf condition cannot declare child conditions",
	},
	{
		// evalOR over zero children is false, so this root can never match.
		name:    "root_or_without_children",
		rules:   "operator: OR\nconditions: []",
		wantErr: "rules: combination condition requires at least one child condition",
	},
	{
		name:    "nested_combination_without_children",
		rules:   "operator: AND\nconditions:\n  - operator: OR\n    conditions: []",
		wantErr: "rules.conditions[0]: combination condition requires at least one child condition",
	},
	{
		name:    "nested_empty_node",
		rules:   "operator: AND\nconditions:\n  - {}",
		wantErr: "rules.conditions[0]: combination condition requires at least one child condition",
	},
}

func TestDecisionRuleTreeShapeCorpus(t *testing.T) {
	runRuleTreeCorpus(t, ruleTreeShapeCorpus)
}

// Every rule-tree error names the decision and the node, including the leaf
// checks that predate the tree walker: in a config with several similar leaves,
// the message is what tells an operator which node to fix.
var ruleTreeErrorLocationCorpus = []ruleTreeCase{
	{
		name: "nested_leaf_unknown_classifier_signal",
		rules: "operator: AND\nconditions:\n  - type: keyword\n    name: k1\n" +
			"  - type: classifier\n    name: nope",
		wantErr: `rules.conditions[1]: signal classifier("nope") is not declared in this recipe`,
	},
	{
		name:    "nested_leaf_on_error_outside_classifier",
		rules:   "operator: AND\nconditions:\n  - type: keyword\n    name: k1\n    on_error: match",
		wantErr: `rules.conditions[0]: condition keyword("k1") on_error is only supported for classifier conditions`,
	},
	{
		name:    "root_leaf_carries_root_path",
		rules:   "type: classifier\nname: nope",
		wantErr: `rules: signal classifier("nope") is not declared in this recipe`,
	},
}

func TestDecisionRuleTreeErrorLocationCorpus(t *testing.T) {
	runRuleTreeCorpus(t, ruleTreeErrorLocationCorpus)
}

// routingFragmentWithRules renders the DSL-owned routing fragment surface, which
// ParseRoutingYAMLBytes loads without the provider/global cross-reference
// validators — including validateDecisionSignalReferences. Rule-tree shape
// therefore has to be enforced by validateDecisionContracts to hold on this path
// too, not only on the full-config path.
func routingFragmentWithRules(rules string) []byte {
	return []byte(`
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
`)
}

// A leaf without a name references no signal at all: evalLeaf looks for a
// matched rule named "", so the decision can never match.
var ruleTreeFragmentCorpus = []ruleTreeCase{
	{
		name:  "leaf_with_name",
		rules: "type: keyword\nname: k1",
	},
	{
		name:    "root_leaf_without_name",
		rules:   "type: keyword",
		wantErr: "rules: leaf condition requires a name",
	},
	{
		name:    "nested_leaf_without_name",
		rules:   "operator: AND\nconditions:\n  - type: keyword",
		wantErr: "rules.conditions[0]: leaf condition requires a name",
	},
	{
		name:  "leaf_type_is_case_insensitive",
		rules: "type: KEYWORD\nname: k1",
	},
	{
		// Projection outputs are intentionally absent from the supported-signal
		// catalog: they are resolved against the projection DAG instead. Reaching
		// that error proves the catalog check exempted the type rather than
		// rejecting it as unsupported.
		name:    "leaf_type_projection_defers_to_the_projection_dag",
		rules:   "type: projection\nname: request_band",
		wantErr: `references projection "request_band"`,
	},
	{
		name:    "leaf_type_unsupported",
		rules:   "type: nonsense\nname: k1",
		wantErr: `rules: unsupported signal type "nonsense"`,
	},
	{
		name:    "nested_leaf_type_unsupported",
		rules:   "operator: AND\nconditions:\n  - type: nonsense\n    name: k1",
		wantErr: `rules.conditions[0]: unsupported signal type "nonsense"`,
	},
	{
		// Nothing trims the type for the evaluator, so a padded spelling is simply
		// not a known signal type.
		name:    "leaf_type_padded",
		rules:   "type: \" keyword\"\nname: k1",
		wantErr: `rules: unsupported signal type " keyword"`,
	},
}

func TestDecisionRuleTreeFragmentCorpus(t *testing.T) {
	for _, tc := range ruleTreeFragmentCorpus {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseRoutingYAMLBytes(routingFragmentWithRules(tc.rules))
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
