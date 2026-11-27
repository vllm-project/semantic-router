package config

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func warnUnguardedSystemOneConditions(decision Decision) {
	if decision.Rules.OnUnknown != "" {
		return
	}
	for _, name := range unguardedSystemOneConditions(&decision.Rules) {
		logging.Warnf(
			"decision '%s': condition %q evaluates as unknown when its model backend fails; set rules.on_unknown or the condition's on_error",
			decision.Name, name,
		)
	}
}

func unguardedSystemOneConditions(node *RuleNode) []string {
	if node == nil {
		return nil
	}
	if node.IsLeaf() {
		if strings.EqualFold(node.Type, SignalTypeSystemOne) && node.OnError == "" {
			return []string{node.Name}
		}
		return nil
	}
	var names []string
	for i := range node.Conditions {
		names = append(names, unguardedSystemOneConditions(&node.Conditions[i])...)
	}
	return names
}
