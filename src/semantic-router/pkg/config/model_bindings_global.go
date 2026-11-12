package config

import (
	"fmt"
)

// GlobalModelScope identifies service-owned handles, not a routable recipe.
const GlobalModelScope RecipeName = "@global"

// EffectiveModelBindings creates an immutable consumer view. Global defaults
// declare execution; a recipe's explicit binding wins only in that recipe.
// Merely declaring a binding never requests that its resource be loaded.
func (c *RouterConfig) EffectiveModelBindings(local map[string]ModelBinding) map[string]ModelBinding {
	bindings := make(map[string]ModelBinding, len(c.GlobalModelBindings)+len(local))
	for name, decl := range c.GlobalModelBindings {
		bindings[name] = decl
	}
	for name, decl := range local {
		bindings[name] = decl
	}
	return bindings
}

// validateGlobalModelBindingContracts checks each global binding: it names a
// declared deployment that answers its application signal.
func validateGlobalModelBindingContracts(cfg *RouterConfig) error {
	for _, name := range sortedModelKeys(cfg.GlobalModelBindings) {
		if err := validateModelBinding(cfg, name, cfg.GlobalModelBindings[name]); err != nil {
			return fmt.Errorf("global.model_catalog.bindings.%s: %w", name, err)
		}
	}
	return nil
}
