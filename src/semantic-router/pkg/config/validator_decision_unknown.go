package config

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func warnUnguardedSystemOneConditions(cfg *RouterConfig, decision Decision) {
	if decision.Rules.OnUnknown != "" {
		return
	}
	for _, name := range unguardedSystemOneConditions(&decision.Rules) {
		when := "when its model backend fails"
		if slices.ContainsFunc(cfg.SystemOneRules, func(rule SystemOneRule) bool { return rule.Name == name && rule.Input.Image }) {
			// Unknown is false by default, so a guard on such a rule needs a policy.
			when += " or the request carries an image no decision model reads; a guard sets rules.on_unknown: match"
		}
		logging.Warnf(
			"decision '%s': condition %q evaluates as unknown %s; set rules.on_unknown or the condition's on_error",
			decision.Name, name, when,
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
