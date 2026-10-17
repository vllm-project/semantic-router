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
	return strings.EqualFold(node.Type, SignalTypeClassifier) || strings.EqualFold(node.Type, SignalTypeDecision)
}

func remoteConditionCanBeUnknown(cfg *RouterConfig, node *RuleNode) bool {
	var bindings []string
	switch strings.ToLower(node.Type) {
	case SignalTypeDomain:
		bindings = []string{"domain_classifier"}
	case SignalTypeComplexity:
		bindings = []string{"complexity"}
	case SignalTypePII:
		if cfg.PIIModel.IsBlock() {
			return false
		}
		bindings = []string{"pii_classifier"}
	case SignalTypeJailbreak:
		if cfg.PromptGuard.IsBlock() {
			return false
		}
		bindings = []string{"prompt_guard"}
	case SignalTypeSafety:
		for _, rule := range cfg.SafetyRules {
			if rule.Name == node.Name && (rule.Model != "" || (rule.Hazard != nil && rule.Hazard.Model != "")) {
				return true
			}
		}
		bindings = []string{"safety." + node.Name, "safety." + node.Name + ".hazard"}
	}
	for _, name := range bindings {
		if decl, bound := cfg.ModelBindings[name]; bound && cfg.ModelDeployments[decl.Deployment].Provider == "http" {
			return true
		}
	}
	return false
}
