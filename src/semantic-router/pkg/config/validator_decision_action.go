package config

import (
	"fmt"
	"strings"
)

func validateDecisionAction(cfg *RouterConfig, decision Decision) error {
	if decision.Action == nil {
		return nil
	}
	if decision.Action.Type != DecisionActionRoute {
		return fmt.Errorf("decision '%s': action.type must be %q", decision.Name, DecisionActionRoute)
	}
	destination := strings.TrimSpace(decision.Action.Destination)
	if destination == "" {
		return fmt.Errorf("decision '%s': action.destination is required", decision.Name)
	}
	if _, ok := cfg.resolveModelConfigKey(destination); !ok {
		return fmt.Errorf("decision '%s': action.destination %q is not defined in model_config", decision.Name, destination)
	}
	if !ruleTreeReferencesSignal(&decision.Rules, SignalTypeJailbreak) {
		return fmt.Errorf("decision '%s': a route action requires an explicit jailbreak condition in rules", decision.Name)
	}
	return nil
}

func ruleTreeReferencesSignal(node *RuleNode, signalType string) bool {
	if node == nil {
		return false
	}
	if strings.EqualFold(node.Type, signalType) {
		return true
	}
	for i := range node.Conditions {
		if ruleTreeReferencesSignal(&node.Conditions[i], signalType) {
			return true
		}
	}
	return false
}
