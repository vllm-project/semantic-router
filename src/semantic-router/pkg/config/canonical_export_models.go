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
		if binding.Head != "" || binding.MappingPath != "" || binding.PairScorer != nil || binding.OperatingPoint != nil {
			return "", "", false
		}
		return binding.Deployment, key, true
	}
	return "", "", false
}

// moveRuleBindings writes on each rule of signals that asks its own question
// the deployment a scope's or the global binding names for it, and drops
// the scope's bindings it wrote.
func moveRuleBindings(signals *CanonicalSignals, scope, global map[string]ModelBinding) {
	used := map[string]bool{}
	take := func(keys ...string) string {
		if name, key, ok := ruleBindingDeployment(scope, keys...); ok {
			used[key] = true
			return name
		}
		name, _, _ := ruleBindingDeployment(global, keys...)
		return name
	}
	for i := range signals.Classifiers {
		rule := &signals.Classifiers[i]
		if rule.Deployment == "" {
			rule.Deployment = take("classifier."+rule.Name, bindingKeyClassifier)
		}
	}
	for i := range signals.Safety {
		rule := &signals.Safety[i]
		if rule.Deployment == "" {
			rule.Deployment = take("safety."+rule.Name, bindingKeySafety)
		}
		if rule.Hazard != nil && rule.Hazard.Deployment == "" {
			hazard := *rule.Hazard
			hazard.Deployment = take("safety."+rule.Name+".hazard", bindingKeyHazard)
			rule.Hazard = &hazard
		}
	}
	for i := range signals.Complexity {
		rule := &signals.Complexity[i]
		if rule.Deployment == "" {
			rule.Deployment = take("complexity."+rule.Name, SignalTypeComplexity)
		}
	}
	for key := range used {
		delete(scope, key)
	}
}

// canonicalizeRuleBindings moves every rule binding a document's scopes and
// global catalog name onto the rules, and drops the global ones for types
// whose rules ask their own question.
func canonicalizeRuleBindings(doc *CanonicalConfig) {
	var global map[string]ModelBinding
	if doc.Global != nil {
		global = doc.Global.ModelCatalog.Bindings
	}
	doc.Routing.Signals.Classifiers = slices.Clone(doc.Routing.Signals.Classifiers)
	doc.Routing.Signals.Safety = slices.Clone(doc.Routing.Signals.Safety)
	doc.Routing.Signals.Complexity = slices.Clone(doc.Routing.Signals.Complexity)
	moveRuleBindings(&doc.Routing.Signals, doc.Routing.ModelBindings, global)
	for i := range doc.Recipes {
		signals := &doc.Recipes[i].Routing.Signals
		signals.Classifiers = slices.Clone(signals.Classifiers)
		signals.Safety = slices.Clone(signals.Safety)
		signals.Complexity = slices.Clone(signals.Complexity)
		moveRuleBindings(signals, doc.Recipes[i].Routing.ModelBindings, global)
	}
	for key := range global {
		_, typeLevel := ruleLevelDeployment[key]
		ruleKey := startsWithAny(key, "classifier.", "safety.", "complexity.")
		if _, _, moved := ruleBindingDeployment(global, key); moved && (typeLevel || ruleKey) {
			delete(global, key)
		}
	}
	if len(doc.Routing.ModelBindings) == 0 {
		doc.Routing.ModelBindings = nil
	}
}
