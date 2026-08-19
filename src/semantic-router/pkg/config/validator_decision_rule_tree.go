package config

import (
	"fmt"
	"slices"
	"strings"
)

// ruleTreeOperators is the closed set of combination operators understood by the
// decision rule evaluator, pkg/decision.(*DecisionEngine).evalNode (and its
// tracing twin in pkg/decision/trace.go). The slice order is the order used in
// error messages.
//
// The evaluator's default branch is OR, so an unrecognized operator never fails
// at runtime: it quietly widens the rule instead. The nested-node half of this
// contract is already enforced by the CLI in src/vllm-sr/cli/models.py
// (Condition.validate_node_shape).
var ruleTreeOperators = []string{"AND", "OR", "NOT"}

// decisionRuleRootPath names a decision's rule-tree root in validation errors.
const decisionRuleRootPath = "rules"

// validateDecisionRuleTree walks a decision's rule tree and reports the failing
// node's location inside that tree, so an error identifies both the decision and
// the offending node.
func validateDecisionRuleTree(cfg *RouterConfig, decisionName, path string, node *RuleNode) error {
	if node == nil {
		return nil
	}
	if node.IsLeaf() {
		return validateDecisionLeafNode(cfg, decisionName, node)
	}
	if err := validateRuleCombinationNode(decisionName, path, node); err != nil {
		return err
	}
	for i := range node.Conditions {
		childPath := fmt.Sprintf("%s.conditions[%d]", path, i)
		if err := validateDecisionRuleTree(cfg, decisionName, childPath, &node.Conditions[i]); err != nil {
			return err
		}
	}
	return nil
}

// validateRuleCombinationNode enforces the contract of a combination node.
//
// The operator is compared without trimming, because every evaluator only
// upper-cases the raw value (pkg/decision/engine.go, pkg/decision/trace.go,
// dashboard topology): a padded " NOT" would pass a trimming check here and then
// fall through to the evaluator's default branch anyway.
func validateRuleCombinationNode(decisionName, path string, node *RuleNode) error {
	operator := strings.ToUpper(node.Operator)
	if operator != "" && !slices.Contains(ruleTreeOperators, operator) {
		return ruleTreeError(decisionName, path, fmt.Sprintf(
			"invalid rule operator %q (valid: %s)",
			node.Operator, strings.Join(ruleTreeOperators, ", ")))
	}
	return nil
}

func ruleTreeError(decisionName, path, message string) error {
	return fmt.Errorf("decision '%s': %s: %s", decisionName, path, message)
}
