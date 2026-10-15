package config

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func warnUnguardedClassifierConditions(decision Decision) {
	if decision.Rules.OnUnknown != "" {
		return
	}
	for _, name := range unguardedClassifierConditions(&decision.Rules) {
		logging.Warnf(
			"decision '%s': condition %q evaluates as unknown when its model backend fails; set rules.on_unknown or the condition's on_error",
			decision.Name, name,
		)
	}
}

func unguardedClassifierConditions(node *RuleNode) []string {
	if node == nil {
		return nil
	}
	if node.IsLeaf() {
		if (strings.EqualFold(node.Type, SignalTypeClassifier) || strings.EqualFold(node.Type, SignalTypeDecision)) && node.OnError == "" {
			return []string{node.Name}
		}
		return nil
	}
	var names []string
	for i := range node.Conditions {
		names = append(names, unguardedClassifierConditions(&node.Conditions[i])...)
	}
	return names
}
