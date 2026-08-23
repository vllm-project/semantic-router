package config

import "fmt"

// visitRoutingProfileConfigs applies a contract to one isolated routing
// profile at a time. It deliberately streams scoped views to the visitor
// instead of building an intermediate profile list.
func visitRoutingProfileConfigs(cfg *RouterConfig, visit func(*RouterConfig) error) error {
	if cfg == nil {
		return visit(cfg)
	}
	if cfg.RoutingScope != "" {
		if err := visit(cfg); err != nil {
			return wrapRoutingProfileValidationError(cfg.RoutingScope, err)
		}
		return nil
	}
	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
		if err := visit(cfg.ConfigForRecipe(recipe)); err != nil {
			return wrapRoutingProfileValidationError(recipe.Name, err)
		}
	}
	for i := range cfg.Entrypoints {
		entrypoint := &cfg.Entrypoints[i]
		for ruleIndex := range entrypoint.Rules {
			rule := &entrypoint.Rules[ruleIndex]
			if rule.derivedRecipe == nil {
				return fmt.Errorf("entrypoint %q rule %q has no compiled action", entrypoint.ID, rule.ID)
			}
			if err := visit(cfg.ConfigForRecipe(rule.derivedRecipe)); err != nil {
				return fmt.Errorf(
					"entrypoint %q rule %q for routing recipe %q: %w",
					entrypoint.ID,
					rule.ID,
					rule.Action.Recipe,
					err,
				)
			}
		}
	}
	return nil
}

func wrapRoutingProfileValidationError(name RecipeName, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("routing recipe %q: %w", name, err)
}
