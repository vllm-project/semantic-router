package config

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func warnUnguardedConditions(cfg *RouterConfig, decision Decision) {
	if decision.Rules.OnUnknown != "" {
		return
	}
	for _, node := range unguardedConditions(cfg, &decision.Rules) {
		fix := "set rules.on_unknown or the condition's on_error"
		if !conditionAcceptsOnError(node) {
			fix = "set rules.on_unknown"
		}
		logging.Warnf(
			"decision '%s': condition %q evaluates as unknown when its model backend fails; %s",
			decision.Name, node.Name, fix,
		)
	}
}

func unguardedConditions(cfg *RouterConfig, node *RuleNode) []*RuleNode {
	if node == nil {
		return nil
	}
	if node.IsLeaf() {
		if node.OnError == "" && (conditionAcceptsOnError(node) || remoteConditionCanBeUnknown(cfg, node)) {
			return []*RuleNode{node}
		}
		return nil
	}
	var nodes []*RuleNode
	for i := range node.Conditions {
		nodes = append(nodes, unguardedConditions(cfg, &node.Conditions[i])...)
	}
	return nodes
}

func conditionAcceptsOnError(node *RuleNode) bool {
	return strings.EqualFold(node.Type, SignalTypeSystemOne)
}

func remoteConditionCanBeUnknown(cfg *RouterConfig, node *RuleNode) bool {
	signalType := strings.ToLower(strings.TrimSpace(node.Type))
	if signalType == SignalTypeJailbreak || signalType == SignalTypePII {
		return false
	}
	rule, ok := cfg.ApplicationRuleByName(signalType, node.Name)
	return ok && cfg.ModelDeployments[cfg.SystemOneRuleDeployment(rule.Rule)].IsRemote()
}
