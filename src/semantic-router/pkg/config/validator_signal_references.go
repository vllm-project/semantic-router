package config

import (
	"fmt"
	"strings"
)

// validateDecisionSignalReferences resolves every rule leaf against the
// selected recipe's declarations. Running this validator on ConfigForRecipe is
// what turns cross-recipe references into deterministic startup errors instead
// of silent non-matches at request time.
func validateDecisionSignalReferences(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	declared := projectionDeclaredSignals(cfg)
	strictReferences := cfg.RoutingScope != ""
	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]
		err := validateRuleNodeSignalReferences(
			decision.Name, decisionRuleRootPath, &decision.Rules, declared, strictReferences)
		if err != nil {
			return err
		}
	}
	return nil
}

func validateRuleNodeSignalReferences(
	decisionName string,
	path string,
	node *RuleNode,
	declared map[string]map[string]struct{},
	strictReferences bool,
) error {
	if node == nil {
		return nil
	}
	if node.IsLeaf() {
		return validateLeafSignalReference(decisionName, path, node, declared, strictReferences)
	}
	for i := range node.Conditions {
		childPath := fmt.Sprintf("%s.conditions[%d]", path, i)
		err := validateRuleNodeSignalReferences(
			decisionName, childPath, &node.Conditions[i], declared, strictReferences)
		if err != nil {
			return err
		}
	}
	return nil
}

func validateLeafSignalReference(
	decisionName string,
	path string,
	node *RuleNode,
	declared map[string]map[string]struct{},
	strictReferences bool,
) error {
	signalType := strings.ToLower(strings.TrimSpace(node.Type))
	name := strings.TrimSpace(node.Name)
	if signalType == SignalTypeProjection {
		// Projection output references are validated after the projection DAG
		// has been checked and its output names are available.
		return nil
	}
	if !IsSupportedSignalType(signalType) {
		return ruleTreeError(decisionName, path, fmt.Sprintf("unsupported signal type %q", node.Type))
	}
	if !strictReferences {
		return nil
	}
	if name == "" {
		return ruleTreeError(decisionName, path, fmt.Sprintf(
			"signal condition of type %q requires a name", signalType))
	}
	if projectionInputDeclared(declared, signalType, name) {
		return nil
	}
	return ruleTreeError(decisionName, path, fmt.Sprintf(
		"signal %s(%q) is not declared in this recipe", signalType, name))
}
