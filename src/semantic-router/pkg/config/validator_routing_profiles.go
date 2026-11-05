package config

import (
	"errors"
	"fmt"
)

// visitRoutingProfileConfigs applies a contract to one isolated routing
// profile at a time. It deliberately streams scoped views to the visitor
// instead of building an intermediate profile list.
func visitRoutingProfileConfigs(cfg *RouterConfig, visit func(*RouterConfig) error) error {
	if cfg == nil || len(cfg.Recipes) == 0 {
		if err := visit(cfg); err != nil {
			return wrapRoutingProfileValidationError(DefaultRecipeName, err)
		}
		return nil
	}

	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
		if err := visit(cfg.ConfigForRecipe(recipe)); err != nil {
			return wrapRoutingProfileValidationError(recipe.Name, err)
		}
	}
	return nil
}

func wrapRoutingProfileValidationError(name RecipeName, err error) error {
	if err == nil {
		return nil
	}
	var located *LocatedError
	if errors.As(err, &located) {
		return err
	}
	return fmt.Errorf("routing recipe %q: %w", name, err)
}
