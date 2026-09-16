package config

import (
	"fmt"
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
	if strings.HasPrefix(name, "classifier.") {
		return classifierSignalRuleByName(signals.ClassifierRules, strings.TrimPrefix(name, "classifier.")) != nil
	}
	if strings.HasPrefix(name, "safety.") {
		for _, rule := range signals.SafetyRules {
			if name == "safety."+rule.Name || (rule.Hazard != nil && name == "safety."+rule.Name+".hazard") {
				return true
			}
		}
		return false
	}
	return true
}

func resolveGlobalModelBindings(cfg *RouterConfig) (map[string]ResolvedModelBinding, error) {
	resolved := make(map[string]ResolvedModelBinding, len(cfg.GlobalModelBindings))
	for _, name := range sortedModelKeys(cfg.GlobalModelBindings) {
		decl := cfg.GlobalModelBindings[name]
		deployment, exists := cfg.ModelDeployments[decl.Deployment]
		if !exists {
			return nil, fmt.Errorf("global.model_catalog.bindings.%s: unknown deployment %q", name, decl.Deployment)
		}
		deployment = deployment.WithDefaults()
		if err := validateTaskModelBinding(name, decl, deployment); err != nil {
			return nil, fmt.Errorf("global.model_catalog.bindings.%s: %w", name, err)
		}
		resolved[name] = ResolvedModelBinding{Recipe: GlobalModelScope, Name: name, Binding: decl, Deployment: deployment, Admission: cfg.ModelAdmission[decl.Deployment]}
	}
	if spec, ok := resolved["embedding"]; ok && cfg.NeedsSemanticResponseCache() {
		model := SemanticCacheEmbeddingModel(cfg)
		primary := strings.ToLower(strings.TrimSpace(cfg.EmbeddingConfig.ModelType))
		if primary == "" {
			primary = "qwen3"
		}
		if model != primary || (spec.Deployment.Provider != "http" && spec.Binding.Adapter != model) {
			return nil, fmt.Errorf("global.stores.response_cache.embedding_model %q must match the global embedding model %q and adapter %q", model, primary, spec.Binding.Adapter)
		}
	}
	return resolved, nil
}

func validateGlobalModelBindingContracts(cfg *RouterConfig) error {
	_, err := resolveGlobalModelBindings(cfg)
	return err
}
