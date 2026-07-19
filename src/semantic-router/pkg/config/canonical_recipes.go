package config

import (
	"fmt"
	"reflect"
	"strings"
)

// CanonicalEntrypoint maps request-facing virtual model names to a named
// recipe in the public v0.3 contract.
type CanonicalEntrypoint struct {
	ModelNames []string `yaml:"model_names"`
	Recipe     string   `yaml:"recipe"`
}

// CanonicalRecipe is a named routing profile selectable through entrypoints.
// Its routing block carries the same profile shape as the top-level `routing`
// block, minus modelCards: the model catalog stays shared across recipes.
type CanonicalRecipe struct {
	Name        string           `yaml:"name"`
	Description string           `yaml:"description,omitempty"`
	Routing     CanonicalRouting `yaml:"routing"`
}

// applyCanonicalRecipeState validates and normalizes `recipes` and
// `entrypoints` into RouterConfig. It runs after applyCanonicalRoutingState,
// so the flat routing fields already hold the top-level routing profile.
func applyCanonicalRecipeState(cfg *RouterConfig, canonical *CanonicalConfig) error {
	if err := validateCanonicalRecipes(canonical); err != nil {
		return err
	}

	recipes := make([]RoutingRecipe, 0, len(canonical.Recipes)+1)
	for _, recipe := range canonical.Recipes {
		decisions := copyDecisions(recipe.Routing.Decisions)
		ensureModelRefDefaults(decisions)
		recipes = append(recipes, RoutingRecipe{
			Name:        recipe.Name,
			Description: recipe.Description,
			Signals:     normalizeSignals(recipe.Routing.Signals, decisions),
			Projections: normalizeProjections(recipe.Routing.Projections),
			Decisions:   decisions,
		})
	}

	if explicitDefault := findRecipe(recipes, DefaultRecipeName); explicitDefault != nil {
		// Recipes-only layout: bridge the explicit default recipe into the
		// flat routing fields so existing read sites keep working.
		cfg.Signals = explicitDefault.Signals
		cfg.Projections = explicitDefault.Projections
		cfg.Decisions = explicitDefault.Decisions
	} else {
		// The top-level routing profile is the default recipe.
		recipes = append([]RoutingRecipe{{
			Name:        DefaultRecipeName,
			Signals:     cfg.Signals,
			Projections: cfg.Projections,
			Decisions:   cfg.Decisions,
		}}, recipes...)
	}
	cfg.Recipes = recipes

	entrypoints, err := normalizeCanonicalEntrypoints(canonical.Entrypoints, recipes)
	if err != nil {
		return err
	}
	cfg.Entrypoints = entrypoints
	return nil
}

func validateCanonicalRecipes(canonical *CanonicalConfig) error {
	modelCards := canonicalRoutingModels(canonical.Routing)
	modelsByName := make(map[string]RoutingModel, len(modelCards))
	for _, model := range modelCards {
		modelsByName[model.Name] = model
	}

	seen := make(map[string]bool, len(canonical.Recipes))
	for _, recipe := range canonical.Recipes {
		name := strings.TrimSpace(recipe.Name)
		if name == "" {
			return fmt.Errorf("recipes[].name cannot be empty")
		}
		if seen[name] {
			return fmt.Errorf("recipes[%s]: duplicate recipe name", name)
		}
		seen[name] = true

		if name == DefaultRecipeName && canonicalRoutingHasProfile(canonical.Routing) {
			return fmt.Errorf("recipes[%s]: conflicts with the top-level routing profile; keep the default profile in `routing` or move it entirely into recipes", name)
		}
		if len(recipe.Routing.ModelCards) > 0 {
			return fmt.Errorf("recipes[%s].routing.modelCards: the model catalog is shared; define modelCards under top-level routing", name)
		}
		if err := validateCanonicalDecisions(recipe.Routing.Decisions, modelsByName, modelCards); err != nil {
			return fmt.Errorf("recipes[%s]: %w", name, err)
		}
	}
	return nil
}

func normalizeCanonicalEntrypoints(entrypoints []CanonicalEntrypoint, recipes []RoutingRecipe) ([]EntrypointMapping, error) {
	if len(entrypoints) == 0 {
		return nil, nil
	}

	result := make([]EntrypointMapping, 0, len(entrypoints))
	claimed := make(map[string]bool)
	for index, entrypoint := range entrypoints {
		recipeName := strings.TrimSpace(entrypoint.Recipe)
		if recipeName == "" {
			return nil, fmt.Errorf("entrypoints[%d].recipe cannot be empty", index)
		}
		if findRecipe(recipes, recipeName) == nil {
			return nil, fmt.Errorf("entrypoints[%d]: unknown recipe %q", index, recipeName)
		}

		names := normalizeAutoModelNames(entrypoint.ModelNames)
		if len(names) == 0 {
			return nil, fmt.Errorf("entrypoints[%d].model_names cannot be empty", index)
		}
		for _, name := range names {
			if claimed[name] {
				return nil, fmt.Errorf("entrypoints[%d]: model name %q is already mapped by another entrypoint", index, name)
			}
			claimed[name] = true
		}

		result = append(result, EntrypointMapping{
			ModelNames: names,
			Recipe:     recipeName,
		})
	}
	return result, nil
}

func findRecipe(recipes []RoutingRecipe, name string) *RoutingRecipe {
	for i := range recipes {
		if recipes[i].Name == name {
			return &recipes[i]
		}
	}
	return nil
}

// canonicalRoutingHasProfile reports whether the routing block carries profile
// content (signals, projections, or decisions). modelCards do not count: they
// are the shared model catalog, not part of any one profile.
func canonicalRoutingHasProfile(routing CanonicalRouting) bool {
	if len(routing.Decisions) > 0 {
		return true
	}
	if len(routing.Projections.Partitions) > 0 || len(routing.Projections.Scores) > 0 || len(routing.Projections.Mappings) > 0 {
		return true
	}
	signals := reflect.ValueOf(routing.Signals)
	for i := 0; i < signals.NumField(); i++ {
		field := signals.Field(i)
		if field.Kind() == reflect.Slice && field.Len() > 0 {
			return true
		}
	}
	return false
}
