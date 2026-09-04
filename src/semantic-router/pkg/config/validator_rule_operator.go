package config

import (
	"fmt"
	"strings"
)

// NormalizeRuleOperator recursively normalizes a rule-tree's composite
// operators in place: it trims whitespace, upper-cases the value, and
// defaults an omitted operator to AND on any node that carries children.
// A node with no type, name, operator, or conditions is left untouched — it
// is the canonical match-all fallback (see RuleNode.IsEmpty) and normalizing
// it would turn a "no rule" root into an explicit empty AND, which the
// decision engine and DSL compiler already special-case identically, but
// which downstream consumers (e.g. the dashboard) use to distinguish "no
// rule was written" from "an empty combinator was written".
//
// This is the single normalization point every producer and evaluator is
// meant to share (issue #2937): once a rule tree has passed through here, an
// omitted composite operator always means AND, case and whitespace variants
// collapse to one canonical form, and any value outside AND/OR/NOT is
// reported as an error naming the offending node instead of being silently
// reinterpreted by whichever consumer reads it next.
func NormalizeRuleOperator(node *RuleNode) error {
	return normalizeRuleOperator(node, "rules")
}

func normalizeRuleOperator(node *RuleNode, path string) error {
	if node == nil || node.IsLeaf() {
		return nil
	}
	if len(node.Conditions) == 0 && node.Operator == "" {
		// Canonical empty node: the decision-root match-all fallback, or an
		// explicitly empty nested placeholder. Nothing to normalize.
		return nil
	}

	op := strings.ToUpper(strings.TrimSpace(node.Operator))
	if op == "" {
		op = "AND"
	}
	switch op {
	case "AND", "OR", "NOT":
		node.Operator = op
	default:
		return fmt.Errorf("%s: unsupported operator %q (expected AND, OR, or NOT)", path, node.Operator)
	}

	for i := range node.Conditions {
		if err := normalizeRuleOperator(&node.Conditions[i], fmt.Sprintf("%s.conditions[%d]", path, i)); err != nil {
			return err
		}
	}
	return nil
}

// validateRuleOperatorContracts normalizes and validates every composite
// rule-tree operator reachable from a routing profile: decision rule trees
// and complexity-signal composer trees. It runs before every other rule-tree
// consuming validator so they all observe the canonical form.
func validateRuleOperatorContracts(cfg *RouterConfig) error {
	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]
		if err := normalizeRuleOperator(&decision.Rules, "rules"); err != nil {
			return fmt.Errorf("decision %q: %w", decision.Name, err)
		}
	}
	for _, rule := range cfg.ComplexityRules {
		if rule.Composer == nil {
			continue
		}
		if err := normalizeRuleOperator(rule.Composer, "composer"); err != nil {
			return fmt.Errorf("complexity rule %q: %w", rule.Name, err)
		}
	}
	return nil
}
