package config

import (
	"errors"
	"fmt"
	"strings"
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
	// A check that names the field it refused writes the field's path in the
	// recipe; the recipe's path leads it, unless the check wrote it already.
	message := err.Error()
	if strings.HasPrefix(message, "recipes[") {
		return err
	}
	for _, field := range []string{"signals.", "projections.", "decisions[", "decisions:", "model_bindings", "plugins", "strategy", "fallback", "candidate_requirements"} {
		if strings.HasPrefix(message, field) {
			return fmt.Errorf("%s.%w", recipePath(string(name)), err)
		}
	}
	return fmt.Errorf("%s: %w", recipePath(string(name)), err)
}

// inRecipe leads an error of a check of the recipe the config is scoped to
// with the recipe's path.
func (c *RouterConfig) inRecipe(err error) error {
	scope := c.RoutingScope
	if scope == "" {
		scope = c.defaultRecipeName()
	}
	return wrapRoutingProfileValidationError(scope, err)
}
