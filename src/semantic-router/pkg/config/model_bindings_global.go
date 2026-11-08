package config

import (
	"fmt"
	"slices"
	"strings"
)

// GlobalModelScope identifies service-owned handles, not a routable recipe.
const GlobalModelScope RecipeName = "@global"

// LookupGlobal resolves only the shared serving catalog. A service must never
// borrow an explicit default- or named-recipe override.
func (p *ModelBindingPlan) LookupGlobal(name string) (ResolvedModelBinding, bool) {
	if p == nil {
		return ResolvedModelBinding{}, false
	}
	spec, ok := p.global[name]
	return spec, ok
}

// EffectiveModelBindings creates an immutable consumer view. Global defaults
// declare execution; a recipe's explicit binding wins only in that recipe.
// Rule-named defaults are visible only where the corresponding rule exists.
// Merely declaring a binding never requests that its resource be loaded.
func (c *RouterConfig) EffectiveModelBindings(signals Signals, local map[string]ModelBinding) map[string]ModelBinding {
	bindings := make(map[string]ModelBinding, len(c.GlobalModelBindings)+len(local))
	for name, decl := range c.GlobalModelBindings {
		if globalBindingApplies(name, signals) {
			bindings[name] = decl
		}
	}
	for name, decl := range local {
		bindings[name] = decl
	}
	return bindings
}

func globalBindingApplies(name string, signals Signals) bool {
	if rule, isRuleKey := strings.CutPrefix(name, "safety."); isRuleKey {
		return slices.ContainsFunc(signals.SafetyRules, func(safety SafetyRule) bool { return safety.Name == rule })
	}
	if rule, isRuleKey := strings.CutPrefix(name, "complexity."); isRuleKey {
		return slices.ContainsFunc(signals.ComplexityRules, func(complexity ComplexityRule) bool { return complexity.Name == rule })
	}
	return true
}

func resolveGlobalModelBindings(cfg *RouterConfig) (map[string]ResolvedModelBinding, error) {
	resolved := make(map[string]ResolvedModelBinding, len(cfg.GlobalModelBindings))
	for _, name := range sortedModelKeys(cfg.GlobalModelBindings) {
		decl := cfg.GlobalModelBindings[name]
		deployment, exists := cfg.ModelDeployments[decl.Deployment]
		if !exists {
			return nil, fmt.Errorf("global.model_catalog.bindings.%s: unknown deployment %q; %s", BindingKey(name), decl.Deployment, declaredDeployments(cfg.ModelDeployments))
		}
		deployment = deployment.WithDefaults()
		if !knownBindingKey(name) {
			return nil, fmt.Errorf("global.model_catalog.bindings.%s: unknown task consumer %q", BindingKey(name), name)
		}
		decl = withImpliedContract(name, decl, deployment)
		if err := validateTaskModelBinding(name, decl, deployment); err != nil {
			return nil, fmt.Errorf("global.model_catalog.bindings.%s: %w", BindingKey(name), err)
		}
		resolved[name] = ResolvedModelBinding{Recipe: GlobalModelScope, Name: name, Binding: decl, Deployment: deployment, Admission: cfg.ModelAdmission[decl.Deployment]}
	}
	return resolved, nil
}

func validateGlobalModelBindingContracts(cfg *RouterConfig) error {
	_, err := resolveGlobalModelBindings(cfg)
	return err
}
