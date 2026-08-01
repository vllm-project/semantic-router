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
		if err := validateRuleNodeSignalReferences(decision.Name, &decision.Rules, declared, strictReferences); err != nil {
			return err
		}
	}
	return nil
}

func validateRuleNodeSignalReferences(
	decisionName string,
	node *RuleNode,
	declared map[string]map[string]struct{},
	strictReferences bool,
) error {
	if node == nil {
		return nil
	}
	if node.IsLeaf() {
		signalType := strings.ToLower(strings.TrimSpace(node.Type))
		name := strings.TrimSpace(node.Name)
		if signalType == SignalTypeProjection {
			// Projection output references are validated after the projection DAG
			// has been checked and its output names are available.
			return nil
		}
		if !IsSupportedSignalType(signalType) {
			return fmt.Errorf("routing.decisions[%q]: unsupported signal type %q", decisionName, node.Type)
		}
		if name == "" {
			if !strictReferences {
				return nil
			}
			return fmt.Errorf("routing.decisions[%q]: signal condition of type %q requires a name", decisionName, signalType)
		}
		if !projectionInputDeclared(declared, signalType, name) {
			if !strictReferences {
				return nil
			}
			return fmt.Errorf(
				"routing.decisions[%q]: signal %s(%q) is not declared in this recipe",
				decisionName,
				signalType,
				name,
			)
		}
		return nil
	}
	for i := range node.Conditions {
		if err := validateRuleNodeSignalReferences(decisionName, &node.Conditions[i], declared, strictReferences); err != nil {
			return err
		}
	}
	return nil
}
