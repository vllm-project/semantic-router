package config

import (
	"fmt"
	"slices"
	"strings"
)

// DefaultRecipeName names the routing profile normalized from the top-level
// `routing:` block. Additional named profiles come from `recipes:`.
const DefaultRecipeName = "default"

// RoutingRecipe is one normalized routing profile. The recipe named
// DefaultRecipeName always mirrors the flat Decisions field on RouterConfig,
// so existing single-profile read sites and recipe-aware read sites observe
// the same default behavior. The flat Signals and Projections fields instead
// hold the global registry: the union of every recipe's profile, so one
// classifier evaluates any recipe's rules (issue #2331 keeps the signal
// registry global).
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

// IsEntrypointModelName reports whether the name is a request-facing virtual
// model name from the entrypoint table. Such names never reach a backend; the
// router resolves them like auto-model aliases.
func (c *RouterConfig) IsEntrypointModelName(modelName string) bool {
	_, ok := c.RecipeForRequestModel(modelName)
	return ok
}

// EntrypointRecipeDescription returns the model-listing description for an
// entrypoint's recipe: the recipe's own description when set, otherwise a
// generic label naming the recipe.
func (c *RouterConfig) EntrypointRecipeDescription(recipeName string) string {
	if recipe, ok := c.RecipeByName(recipeName); ok && strings.TrimSpace(recipe.Description) != "" {
		return recipe.Description
	}
	return fmt.Sprintf("Entrypoint for the %s routing recipe", recipeName)
}

// AllRoutingDecisions returns the decisions of every routing profile, for
// callers that reason about routing as a whole (signal usage analysis,
// contract validation). Configs built without the canonical loader carry no
// recipes; their flat decisions are the only profile.
func (c *RouterConfig) AllRoutingDecisions() []Decision {
	if c == nil {
		return nil
	}
	if len(c.Recipes) == 0 {
		return c.Decisions
	}
	if len(c.Recipes) == 1 {
		return c.Recipes[0].Decisions
	}
	all := make([]Decision, 0, 2*len(c.Decisions))
	for i := range c.Recipes {
		all = append(all, c.Recipes[i].Decisions...)
	}
	return all
}

// RoutingProfileSignals returns the default profile's signals for canonical
// export. The flat Signals field holds the global registry (union across
// recipes); exporting it would leak other recipes' rules into the top-level
// routing block.
func (c *RouterConfig) RoutingProfileSignals() Signals {
	if c == nil {
		return Signals{}
	}
	if recipe := c.DefaultRecipe(); recipe != nil {
		return recipe.Signals
	}
	return c.Signals
}

// RoutingProfileProjections is the projections counterpart of
// RoutingProfileSignals.
func (c *RouterConfig) RoutingProfileProjections() Projections {
	if c == nil {
		return Projections{}
	}
	if recipe := c.DefaultRecipe(); recipe != nil {
		return recipe.Projections
	}
	return c.Projections
}
