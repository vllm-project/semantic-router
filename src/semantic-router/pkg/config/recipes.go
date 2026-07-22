package config

import (
	"slices"
	"strings"
)

// DefaultRecipeName names the routing profile normalized from the top-level
// `routing:` block. Additional named profiles come from `recipes:`.
const DefaultRecipeName = "default"

// RoutingRecipe is one normalized routing profile. The recipe named
// DefaultRecipeName always mirrors the flat routing fields on RouterConfig,
// so existing single-profile read sites and recipe-aware read sites observe
// the same default behavior.
type RoutingRecipe struct {
	Name        string
	Description string
	Signals     Signals
	Projections Projections
	Decisions   []Decision
}

// EntrypointMapping binds request-facing virtual model names to a named
// recipe. The virtual names never reach a backend; they only select which
// routing profile evaluates the request.
type EntrypointMapping struct {
	ModelNames []string
	Recipe     string
}

// RecipeByName returns the normalized recipe with the given name.
func (c *RouterConfig) RecipeByName(name string) (*RoutingRecipe, bool) {
	if c == nil {
		return nil, false
	}
	for i := range c.Recipes {
		if c.Recipes[i].Name == name {
			return &c.Recipes[i], true
		}
	}
	return nil, false
}

// DefaultRecipe returns the recipe backing the flat routing fields, or nil
// for configs built without the canonical loader (for example DSL fragments).
func (c *RouterConfig) DefaultRecipe() *RoutingRecipe {
	recipe, ok := c.RecipeByName(DefaultRecipeName)
	if !ok {
		return nil
	}
	return recipe
}

// RecipeForRequestModel resolves a request model name through the entrypoint
// table. It returns false when the name matches no entrypoint; callers fall
// back to auto-model or specified-model handling.
func (c *RouterConfig) RecipeForRequestModel(modelName string) (*RoutingRecipe, bool) {
	if c == nil {
		return nil, false
	}
	trimmed := strings.TrimSpace(modelName)
	if trimmed == "" {
		return nil, false
	}
	for _, entrypoint := range c.Entrypoints {
		if slices.Contains(entrypoint.ModelNames, trimmed) {
			return c.RecipeByName(entrypoint.Recipe)
		}
	}
	return nil, false
}
