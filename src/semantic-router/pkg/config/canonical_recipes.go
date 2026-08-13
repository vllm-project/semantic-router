package config

import (
	"fmt"
	"reflect"
	"strings"
)

// CanonicalEntrypoint maps request-facing virtual model names to a named
// recipe in the public v0.3 contract. Exactly one of Recipe (legacy,
// unconditional) or Rules (conditional, evaluated per-caller) is set — see
// EntrypointMapping's doc comment (recipes.go) for the full contract.
type CanonicalEntrypoint struct {
	ModelNames []string                  `yaml:"model_names"`
	Recipe     string                    `yaml:"recipe,omitempty"`
	Rules      []CanonicalEntrypointRule `yaml:"rules,omitempty"`
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
		strategy := recipe.Routing.Strategy
		if strategy == "" {
			strategy = cfg.Strategy
		}
		recipes = append(recipes, RoutingRecipe{
			Name:        RecipeName(recipe.Name),
			Description: recipe.Description,
			Profile: RoutingProfile{
				Signals:     normalizeSignals(recipe.Routing.Signals, decisions),
				Projections: normalizeProjections(recipe.Routing.Projections),
				Decisions:   decisions,
				Strategy:    strategy,
			},
		})
	}

	if explicitDefault := findRecipe(recipes, DefaultRecipeName); explicitDefault != nil {
		// Recipes-only layout: bridge the explicit default recipe into the
		// flat routing fields so existing single-profile read sites keep working.
		cfg.Signals = explicitDefault.Profile.Signals
		cfg.Projections = explicitDefault.Profile.Projections
		cfg.Decisions = explicitDefault.Profile.Decisions
		cfg.Strategy = explicitDefault.Profile.Strategy
	} else {
		// The top-level routing profile is the default recipe.
		recipes = append([]RoutingRecipe{{
			Name: DefaultRecipeName,
			Profile: RoutingProfile{
				Signals:     cfg.Signals,
				Projections: cfg.Projections,
				Decisions:   cfg.Decisions,
				Strategy:    cfg.Strategy,
			},
		}}, recipes...)
	}
	cfg.Recipes = recipes

	entrypoints, err := normalizeCanonicalEntrypoints(cfg, canonical, recipes)
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

	seen := make(map[RecipeName]struct{}, len(canonical.Recipes))
	for _, recipe := range canonical.Recipes {
		name := RecipeName(strings.TrimSpace(recipe.Name))
		if name == "" {
			return fmt.Errorf("recipes[].name cannot be empty")
		}
		if string(name) != recipe.Name {
			return fmt.Errorf(
				"recipes[%s].name must not contain surrounding whitespace",
				name,
			)
		}
		if _, exists := seen[name]; exists {
			return fmt.Errorf("recipes[%s]: duplicate recipe name", name)
		}
		seen[name] = struct{}{}

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

func normalizeCanonicalEntrypoints(cfg *RouterConfig, canonical *CanonicalConfig, recipes []RoutingRecipe) ([]EntrypointMapping, error) {
	entrypoints := canonical.Entrypoints
	if len(entrypoints) == 0 {
		return nil, nil
	}

	result := make([]EntrypointMapping, 0, len(entrypoints))
	claimed := make(map[string]struct{})
	for index, entrypoint := range entrypoints {
		names := normalizeAutoModelNames(entrypoint.ModelNames)
		if len(names) == 0 {
			return nil, fmt.Errorf("entrypoints[%d].model_names cannot be empty", index)
		}
		for _, name := range names {
			if _, exists := claimed[name]; exists {
				return nil, fmt.Errorf("entrypoints[%d]: model name %q is already mapped by another entrypoint", index, name)
			}
			claimed[name] = struct{}{}
			if meaning := entrypointNameConflict(cfg, canonical, name); meaning != "" {
				return nil, fmt.Errorf("entrypoints[%d]: model name %q is already %s; entrypoint names must be new virtual names", index, name, meaning)
			}
		}

		mapping := EntrypointMapping{ModelNames: names}
		hasRecipe := strings.TrimSpace(entrypoint.Recipe) != ""
		hasRules := len(entrypoint.Rules) > 0
		switch {
		case hasRecipe && hasRules:
			return nil, fmt.Errorf("entrypoints[%d]: set either recipe or rules, not both", index)
		case !hasRecipe && !hasRules:
			return nil, fmt.Errorf("entrypoints[%d]: recipe cannot be empty (or set rules for a conditional entrypoint)", index)
		case hasRecipe:
			// Legacy, unconditional form: unchanged from the pre-rules contract.
			recipeName := RecipeName(strings.TrimSpace(entrypoint.Recipe))
			if findRecipe(recipes, recipeName) == nil {
				return nil, fmt.Errorf("entrypoints[%d]: unknown recipe %q", index, recipeName)
			}
			mapping.Recipe = recipeName
		default:
			rules, err := normalizeCanonicalEntrypointRules(entrypoint.Rules, recipes, index)
			if err != nil {
				return nil, err
			}
			mapping.Rules = rules
		}

		result = append(result, mapping)
	}
	return result, nil
}

// entrypointNameConflict reports what an entrypoint virtual name already means
// to the router, if anything. Entrypoint names must be new: reusing an existing
// routable name would silently hijack it, because requestModelActsAsAuto stops
// treating the name as an explicitly specified model.
func entrypointNameConflict(cfg *RouterConfig, canonical *CanonicalConfig, name string) string {
	for _, model := range canonicalRoutingModels(canonical.Routing) {
		if model.Name == name {
			return "a configured model"
		}
		if routingModelHasLoRA(model, name) {
			return "a configured LoRA adapter"
		}
	}
	switch {
	case cfg.IsAutoModelName(name):
		return "an auto-model alias"
	case cfg.IsReMoMModelName(name):
		return "the ReMoM algorithm slug"
	case cfg.IsFusionModelName(name):
		return "the Fusion algorithm slug"
	case cfg.IsFlowModelName(name):
		return "the Flow algorithm slug"
	}
	return ""
}

// canonicalRecipesFromRouterConfig exports the normalized named recipes. The
// default recipe is not exported here: it round-trips as the top-level
// routing block.
func canonicalRecipesFromRouterConfig(cfg *RouterConfig) []CanonicalRecipe {
	if cfg == nil || len(cfg.Recipes) == 0 {
		return nil
	}
	recipes := make([]CanonicalRecipe, 0, len(cfg.Recipes))
	for _, recipe := range cfg.Recipes {
		if recipe.Name == DefaultRecipeName {
			continue
		}
		recipes = append(recipes, CanonicalRecipe{
			Name:        string(recipe.Name),
			Description: recipe.Description,
			Routing: CanonicalRouting{
				Signals:     canonicalSignalsFromSignals(recipe.Profile.Signals),
				Projections: canonicalProjectionsFromProjections(recipe.Profile.Projections),
				Decisions:   copyDecisions(recipe.Profile.Decisions),
				Strategy:    recipe.Profile.Strategy,
			},
		})
	}
	if len(recipes) == 0 {
		return nil
	}
	return recipes
}

// canonicalEntrypointsFromRouterConfig exports the normalized entrypoint table.
func canonicalEntrypointsFromRouterConfig(cfg *RouterConfig) []CanonicalEntrypoint {
	if cfg == nil || len(cfg.Entrypoints) == 0 {
		return nil
	}
	entrypoints := make([]CanonicalEntrypoint, 0, len(cfg.Entrypoints))
	for _, entrypoint := range cfg.Entrypoints {
		entrypoints = append(entrypoints, CanonicalEntrypoint{
			ModelNames: append([]string(nil), entrypoint.ModelNames...),
			Recipe:     string(entrypoint.Recipe),
			Rules:      canonicalEntrypointRulesFromNormalized(entrypoint.Rules),
		})
	}
	return entrypoints
}

func findRecipe(recipes []RoutingRecipe, name RecipeName) *RoutingRecipe {
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
	if routing.Strategy != "" {
		return true
	}
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
