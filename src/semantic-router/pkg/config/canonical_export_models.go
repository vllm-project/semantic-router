package config

import (
	"slices"
	"strings"
)

// The export writes model resources in the form a configuration writes:
// a deployment's served model as model and its question types under
// capabilities, and the deployment of a rule that asks its own question on
// the rule, wherever a binding (one a deprecated field declared, or a
// binding for one rule) named it.

func startsWithAny(key string, prefixes ...string) bool {
	for _, prefix := range prefixes {
		if len(key) > len(prefix) && strings.HasPrefix(key, prefix) {
			return true
		}
	}
	return false
}

// canonicalDeployments writes deployments in their canonical form.
func canonicalDeployments(deployments map[string]ModelDeployment) map[string]ModelDeployment {
	out := cloneModelMap(deployments)
	for name, deployment := range out {
		if deployment.Model == "" {
			deployment.Model = deployment.ServedName
		}
		deployment.ServedName = ""
		if len(deployment.QuestionTypes) > 0 {
			capabilities := DeploymentCapabilities{}
			if deployment.Capabilities != nil {
				capabilities = *deployment.Capabilities
			}
			if len(capabilities.QuestionTypes) == 0 {
				capabilities.QuestionTypes = slices.Clone(deployment.QuestionTypes)
			}
			deployment.Capabilities, deployment.QuestionTypes = &capabilities, nil
		}
		deployment.Replicas = slices.Clone(deployment.Replicas)
		for i := range deployment.Replicas {
			if deployment.Replicas[i].Model == "" {
				deployment.Replicas[i].Model = deployment.Replicas[i].ServedName
			}
			deployment.Replicas[i].ServedName = ""
		}
		out[name] = deployment
	}
	return out
}

// ruleBindingDeployment is the deployment a binding names for a rule when
// the binding carries nothing a rule's deployment cannot.
func ruleBindingDeployment(bindings map[string]ModelBinding, keys ...string) (string, string, bool) {
	for _, key := range keys {
		binding, ok := bindings[key]
		if !ok {
			continue
		}
		if binding.Head != "" || binding.MappingPath != "" || binding.OperatingPoint != nil {
			return "", "", false
		}
		return binding.Deployment, key, true
	}
	return "", "", false
}

// moveRuleBindings writes on each safety and complexity rule the deployment
// a binding for that one rule (a deprecated safety.<rule> or
// complexity.<rule> key) names in the scope or globally, and drops the
// scope's bindings it wrote.
func moveRuleBindings(signals *CanonicalSignals, scope, global map[string]ModelBinding) {
	used := map[string]bool{}
	take := func(key string) string {
		if name, scopeKey, ok := ruleBindingDeployment(scope, key); ok {
			used[scopeKey] = true
			return name
		}
		name, _, _ := ruleBindingDeployment(global, key)
		return name
	}
	for i := range signals.Safety {
		rule := &signals.Safety[i]
		if rule.Deployment == "" {
			rule.Deployment = take("safety." + rule.Name)
		}
	}
	for i := range signals.Complexity {
		rule := &signals.Complexity[i]
		if rule.Deployment == "" {
			rule.Deployment = take("complexity." + rule.Name)
		}
	}
	for key := range used {
		delete(scope, key)
	}
}

// canonicalizeRuleBindings moves every binding for one rule a document's
// scopes and global catalog name onto the rule.
func canonicalizeRuleBindings(doc *CanonicalConfig) {
	var global map[string]ModelBinding
	if doc.Global != nil {
		global = doc.Global.ModelCatalog.Bindings
	}
	for i := range doc.Recipes {
		signals := &doc.Recipes[i].Signals
		signals.Safety = slices.Clone(signals.Safety)
		signals.Complexity = slices.Clone(signals.Complexity)
		moveRuleBindings(signals, doc.Recipes[i].ModelBindings, global)
		if len(doc.Recipes[i].ModelBindings) == 0 {
			doc.Recipes[i].ModelBindings = nil
		}
	}
	for key := range global {
		if _, _, moved := ruleBindingDeployment(global, key); moved && startsWithAny(key, "safety.", "complexity.") {
			delete(global, key)
		}
	}
}
