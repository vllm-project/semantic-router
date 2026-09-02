package config

import (
	"fmt"
	"strings"
)

// Rule-tree combination operators. These are the only values a normalized
// RuleNode.Operator can hold; every evaluator switches on them so the
// validator and the runtime share one operator set.
const (
	RuleOperatorAnd = "AND"
	RuleOperatorOr  = "OR"
	RuleOperatorNot = "NOT"
)

// RuleTreeOperators lists the supported combination operators in the order
// they are documented.
func RuleTreeOperators() []string {
	return []string{RuleOperatorAnd, RuleOperatorOr, RuleOperatorNot}
}

// IsRuleTreeOperator reports whether op is a supported, already-normalized
// combination operator.
func IsRuleTreeOperator(op string) bool {
	switch op {
	case RuleOperatorAnd, RuleOperatorOr, RuleOperatorNot:
		return true
	}
	return false
}

// NormalizeRuleOperator recursively normalizes a rule-tree's composite
// operators in place and validates the shape of every node: it trims
// whitespace, upper-cases the value, and defaults an omitted operator to AND
// on any node that carries children. A root node with no type, name,
// operator, or conditions is left untouched — it is the canonical match-all
// fallback (see RuleNode.IsEmpty) and normalizing it would turn a "no rule"
// root into an explicit empty AND, which the decision engine and DSL compiler
// already special-case identically, but which downstream consumers (e.g. the
// dashboard) use to distinguish "no rule was written" from "an empty
// combinator was written".
//
// This is the single normalization point every producer and evaluator is
// meant to share (issues #2122 and #2937): once a rule tree has passed
// through here, an omitted composite operator always means AND, case and
// whitespace variants collapse to one canonical form, NOT carries exactly one
// child, every node is unambiguously a leaf or a combination, and any value
// outside AND/OR/NOT is reported as an error naming the offending node
// instead of being silently reinterpreted by whichever consumer reads it
// next.
func NormalizeRuleOperator(node *RuleNode) error {
	return normalizeRuleOperator(node, "rules", true)
}

func normalizeRuleOperator(node *RuleNode, path string, root bool) error {
	if node == nil {
		return nil
	}
	if err := validateRuleNodeShape(node, path); err != nil {
		return err
	}
	if node.IsLeaf() {
		return nil
	}
	if len(node.Conditions) == 0 && node.Operator == "" {
		if root {
			// Canonical empty root: the decision-level match-all fallback.
			return nil
		}
		return fmt.Errorf("%s: combination condition requires an operator and at least one child condition", path)
	}

	op, err := normalizeCombinationOperator(node, path)
	if err != nil {
		return err
	}
	if err := validateCombinationArity(op, len(node.Conditions), path, root); err != nil {
		return err
	}

	for i := range node.Conditions {
		if err := normalizeRuleOperator(&node.Conditions[i], fmt.Sprintf("%s.conditions[%d]", path, i), false); err != nil {
			return err
		}
	}
	return nil
}

// normalizeCombinationOperator collapses case and whitespace variants of a
// combination operator to the canonical form, defaults an omitted operator to
// AND, and rejects anything outside the supported set.
func normalizeCombinationOperator(node *RuleNode, path string) (string, error) {
	op := strings.ToUpper(strings.TrimSpace(node.Operator))
	if op == "" {
		op = RuleOperatorAnd
	}
	if !IsRuleTreeOperator(op) {
		return "", fmt.Errorf("%s: unsupported operator %q (expected %s)",
			path, node.Operator, strings.Join(RuleTreeOperators(), ", "))
	}
	node.Operator = op
	return op, nil
}

// validateCombinationArity enforces the child-count contract of a combination
// node: NOT is strictly unary, and only a root AND may be childless.
func validateCombinationArity(op string, children int, path string, root bool) error {
	if op == RuleOperatorNot && children != 1 {
		return fmt.Errorf("%s: NOT requires exactly one child condition, got %d", path, children)
	}
	if children == 0 && !isMatchAllCombination(op, root) {
		return fmt.Errorf("%s: %s combination requires at least one child condition", path, op)
	}
	return nil
}

// validateRuleNodeShape rejects nodes that are neither a well-formed leaf nor
// a well-formed combination.
//
// IsLeaf keys off `type` alone, so a node that carries other leaf fields
// without one is silently treated as a combination and loses what the author
// wrote: with `name` set it evaluates as a childless combination that never
// matches, and with only `label`, `predicate` or `on_error` set it also
// satisfies IsEmpty at a decision root, which the engine short-circuits into an
// unconditional match. A leaf that also declares an operator or children is
// ambiguous: the evaluators would ignore the combination half entirely.
func validateRuleNodeShape(node *RuleNode, path string) error {
	hasLeafFields := node.Name != "" || node.Label != "" || node.Predicate != nil || node.OnError != ""
	if node.Type == "" {
		if !hasLeafFields {
			return nil
		}
		return fmt.Errorf("%s: leaf condition requires a type (name, label, predicate, and on_error only apply to leaf conditions)", path)
	}
	if node.Operator != "" || len(node.Conditions) > 0 {
		return fmt.Errorf("%s: condition must be either a leaf (type/name) or a combination (operator/conditions), not both", path)
	}
	return nil
}

// isMatchAllCombination reports whether a childless combination node is the
// canonical unconditional-match shape. Only a tree's root qualifies, and only
// for AND, which is the one operator that matches on zero children: a
// childless OR evaluates to false, i.e. a decision that can never match, and
// a childless NOT is already rejected by the arity check.
func isMatchAllCombination(op string, root bool) bool {
	return root && op == RuleOperatorAnd
}

// validateRuleOperatorContracts normalizes and validates every composite
// rule-tree operator reachable from a routing profile: decision rule trees
// and complexity-signal composer trees. It runs before every other rule-tree
// consuming validator so they all observe the canonical form.
func validateRuleOperatorContracts(cfg *RouterConfig) error {
	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]
		if err := normalizeRuleOperator(&decision.Rules, "rules", true); err != nil {
			return fmt.Errorf("decision %q: %w", decision.Name, err)
		}
	}
	for _, rule := range cfg.ComplexityRules {
		if rule.Composer == nil {
			continue
		}
		if err := normalizeRuleOperator(rule.Composer, "composer", true); err != nil {
			return fmt.Errorf("complexity rule %q: %w", rule.Name, err)
		}
	}
	return nil
}
