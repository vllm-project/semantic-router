package config

import (
	"fmt"
	"slices"
	"strings"
)

// Rule-tree combination operators. They are exported so the evaluators can be
// pinned to the same set instead of restating it as literals: config validation
// rejects everything outside RuleTreeOperators, which is only safe while every
// member reaches its own branch in pkg/decision.(*DecisionEngine).evalNode. That
// agreement is asserted by TestRuleTreeOperatorsAgreeWithEvaluator.
const (
	// RuleOperatorAnd matches when every condition matches. It is also the only
	// operator that matches on zero children.
	RuleOperatorAnd = "AND"
	// RuleOperatorOr matches when at least one condition matches. It is the
	// evaluator's default branch, so an unrecognized operator quietly widens the
	// rule instead of failing at runtime.
	RuleOperatorOr = "OR"
	// RuleOperatorNot is strictly unary: evalNOT negates its single child and
	// warns and reports a non-match for any other child count.
	RuleOperatorNot = "NOT"
)

// RuleTreeOperators is the closed set of combination operators a decision rule
// tree may use. The slice order is the order used in error messages. The
// nested-node half of this contract is already enforced by the CLI in
// src/vllm-sr/cli/models.py (Condition.validate_node_shape).
var RuleTreeOperators = []string{RuleOperatorAnd, RuleOperatorOr, RuleOperatorNot}

// decisionRuleRootPath names a decision's rule-tree root in validation errors.
const decisionRuleRootPath = "rules"

// validateDecisionRuleTree walks a decision's rule tree and reports the failing
// node's location inside that tree, so an error identifies both the decision and
// the offending node.
func validateDecisionRuleTree(cfg *RouterConfig, decisionName, path string, node *RuleNode, isRoot bool) error {
	if node == nil {
		return nil
	}
	if err := validateRuleNodeShape(decisionName, path, node); err != nil {
		return err
	}
	if node.IsLeaf() {
		return validateDecisionLeafNode(cfg, decisionName, path, node)
	}
	if err := validateRuleCombinationNode(decisionName, path, node, isRoot); err != nil {
		return err
	}
	for i := range node.Conditions {
		childPath := fmt.Sprintf("%s.conditions[%d]", path, i)
		if err := validateDecisionRuleTree(cfg, decisionName, childPath, &node.Conditions[i], false); err != nil {
			return err
		}
	}
	return nil
}

// validateRuleNodeShape rejects nodes that are neither a well-formed leaf nor a
// well-formed combination.
//
// IsLeaf keys off `type` alone, so a node that carries leaf fields without one
// is treated as a combination and loses what the author wrote: with `name` set
// it evaluates as an empty OR, a decision that can never match, and with only
// `label`, `predicate` or `on_error` set it also satisfies IsEmpty, which the
// engine short-circuits into an unconditional match.
func validateRuleNodeShape(decisionName, path string, node *RuleNode) error {
	if !ruleNodeHasLeafFields(node) {
		return nil
	}
	if node.Operator != "" {
		return ruleTreeError(decisionName, path,
			"node must be either a leaf (type/name) or a combination (operator/conditions), not both")
	}
	if node.Type == "" {
		return ruleTreeError(decisionName, path, "leaf condition requires a type")
	}
	if len(node.Conditions) > 0 {
		return ruleTreeError(decisionName, path, "leaf condition cannot declare child conditions")
	}
	return nil
}

func ruleNodeHasLeafFields(node *RuleNode) bool {
	return node.Type != "" || node.Name != "" || node.Label != "" ||
		node.Predicate != nil || node.OnError != ""
}

// validateRuleCombinationNode enforces the contract of a combination node.
//
// The operator is compared without trimming, because every evaluator only
// upper-cases the raw value (pkg/decision/engine.go, pkg/decision/trace.go,
// dashboard topology): a padded " NOT" would pass a trimming check here and then
// fall through to the evaluator's default branch anyway.
func validateRuleCombinationNode(decisionName, path string, node *RuleNode, isRoot bool) error {
	operator := strings.ToUpper(node.Operator)
	if operator != "" && !slices.Contains(RuleTreeOperators, operator) {
		return ruleTreeError(decisionName, path, fmt.Sprintf(
			"invalid rule operator %q (valid: %s)",
			node.Operator, strings.Join(RuleTreeOperators, ", ")))
	}
	if operator == RuleOperatorNot && len(node.Conditions) != 1 {
		return ruleTreeError(decisionName, path, fmt.Sprintf(
			"NOT requires exactly one child condition, got %d", len(node.Conditions)))
	}
	if len(node.Conditions) == 0 && !isMatchAllCombination(operator, isRoot) {
		return ruleTreeError(decisionName, path,
			"combination condition requires at least one child condition")
	}
	return nil
}

// isMatchAllCombination reports whether a childless combination node is the
// canonical unconditional-match shape. Only a decision's root qualifies, and
// only for the operators that actually match on zero children: an omitted
// operator, which IsEmpty short-circuits to a match, and AND. A childless OR
// evaluates to false, i.e. a decision that can never match.
func isMatchAllCombination(operator string, isRoot bool) bool {
	return isRoot && (operator == "" || operator == RuleOperatorAnd)
}

func ruleTreeError(decisionName, path, message string) error {
	return fmt.Errorf("decision '%s': %s: %s", decisionName, path, message)
}
