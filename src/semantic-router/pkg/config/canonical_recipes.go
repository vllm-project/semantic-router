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
		// flat decisions so existing single-profile read sites keep working.
		cfg.DefaultDecisions = explicitDefault.Decisions
	} else {
		// The top-level routing profile is the default recipe.
		recipes = append([]RoutingRecipe{{
			Name:        DefaultRecipeName,
			Signals:     cfg.Signals,
			Projections: cfg.Projections,
			Decisions:   cfg.DefaultDecisions,
		}}, recipes...)
	}
	cfg.Recipes = recipes

	// Decision names key response headers, metrics, and per-decision plugin
	// lookups (GetDecisionByName), so they share one global namespace.
	if err := validateDecisionNamesAcrossRecipes(recipes); err != nil {
		return err
	}

	// The signal registry is global (#2331): every recipe's signals and
	// projections merge into the flat fields so one classifier evaluates any
	// recipe's rules, while decisions stay per-recipe.
	signals, projections, err := mergeRecipeRegistries(recipes)
	if err != nil {
		return err
	}
	cfg.Signals = signals
	cfg.Projections = projections

	entrypoints, err := normalizeCanonicalEntrypoints(cfg, canonical, recipes)
	if err != nil {
		return err
	}
	cfg.Entrypoints = entrypoints
	return nil
}

// validateDecisionNamesAcrossRecipes rejects a decision name declared by more
// than one profile. Everything that keys on the bare decision name at runtime
// (x-vsr-selected-decision, metrics, plugin lookups) would silently attribute
// one recipe's decision to another's.
func validateDecisionNamesAcrossRecipes(recipes []RoutingRecipe) error {
	owners := make(map[string]string)
	for _, recipe := range recipes {
		for _, decision := range recipe.Decisions {
			owner, exists := owners[decision.Name]
			if !exists {
				owners[decision.Name] = recipe.Name
				continue
			}
			if owner == recipe.Name {
				return fmt.Errorf("recipes[%s]: duplicate decision name %q", recipe.Name, decision.Name)
			}
			return fmt.Errorf(
				"recipes: decision %q is defined by both the %q and %q profiles; decision names are global (headers, metrics, plugin lookups), rename one",
				decision.Name, owner, recipe.Name,
			)
		}
	}
	return nil
}

// mergeRecipeRegistries builds the global signal and projection registries
// from every recipe's profile. Single-recipe configs pass through untouched so
// single-profile behavior cannot change.
func mergeRecipeRegistries(recipes []RoutingRecipe) (Signals, Projections, error) {
	if len(recipes) == 1 {
		return recipes[0].Signals, recipes[0].Projections, nil
	}
	signals, err := mergeRecipeSignals(recipes)
	if err != nil {
		return Signals{}, Projections{}, err
	}
	projections, err := mergeRecipeProjections(recipes)
	if err != nil {
		return Signals{}, Projections{}, err
	}
	return signals, projections, nil
}

// mergeRecipeSignals appends every recipe's rules field by field. Rule names
// share one global namespace per signal kind, so a name declared by two
// profiles with different content is ambiguous and rejected. Identical
// redeclarations merge silently: profiles that route on the same domain name
// without declaring signals.domains each auto-generate the same placeholder
// category (normalizeSignals), and that must not fail a config that loads on
// a single profile.
func mergeRecipeSignals(recipes []RoutingRecipe) (Signals, error) {
	var merged Signals
	mergedValue := reflect.ValueOf(&merged).Elem()
	owners := make(map[string]string)
	claimedRules := make(map[string]reflect.Value)
	for _, recipe := range recipes {
		recipeValue := reflect.ValueOf(recipe.Signals)
		for i := 0; i < mergedValue.NumField(); i++ {
			kind := strings.Split(mergedValue.Type().Field(i).Tag.Get("yaml"), ",")[0]
			rules := recipeValue.Field(i)
			for j := 0; j < rules.Len(); j++ {
				rule := rules.Index(j)
				name := rule.FieldByName("Name").String()
				key := kind + ":" + name
				if prior, exists := claimedRules[key]; exists && reflect.DeepEqual(prior.Interface(), rule.Interface()) {
					continue
				}
				if err := claimRegistryName(owners, kind, name, recipe.Name, "signal"); err != nil {
					return Signals{}, err
				}
				claimedRules[key] = rule
				mergedValue.Field(i).Set(reflect.Append(mergedValue.Field(i), rule))
			}
		}
	}
	return merged, nil
}

// mergeRecipeProjections is the projections counterpart of mergeRecipeSignals.
func mergeRecipeProjections(recipes []RoutingRecipe) (Projections, error) {
	var merged Projections
	owners := make(map[string]string)
	for _, recipe := range recipes {
		for _, partition := range recipe.Projections.Partitions {
			if err := claimRegistryName(owners, "partitions", partition.Name, recipe.Name, "projection"); err != nil {
				return Projections{}, err
			}
			merged.Partitions = append(merged.Partitions, partition)
		}
		for _, score := range recipe.Projections.Scores {
			if err := claimRegistryName(owners, "scores", score.Name, recipe.Name, "projection"); err != nil {
				return Projections{}, err
			}
			merged.Scores = append(merged.Scores, score)
		}
		for _, mapping := range recipe.Projections.Mappings {
			if err := claimRegistryName(owners, "mappings", mapping.Name, recipe.Name, "projection"); err != nil {
				return Projections{}, err
			}
			merged.Mappings = append(merged.Mappings, mapping)
		}
	}
	return merged, nil
}

func claimRegistryName(owners map[string]string, kind, name, recipeName, registry string) error {
	key := kind + ":" + name
	if owner, exists := owners[key]; exists {
		return fmt.Errorf(
			"recipes: %s %s %q is defined by both the %q and %q profiles; the %s registry is global, define it once",
			registry, kind, name, owner, recipeName, registry,
		)
	}
	owners[key] = recipeName
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

func normalizeCanonicalEntrypoints(cfg *RouterConfig, canonical *CanonicalConfig, recipes []RoutingRecipe) ([]EntrypointMapping, error) {
	entrypoints := canonical.Entrypoints
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
			if meaning := entrypointNameConflict(cfg, canonical, name); meaning != "" {
				return nil, fmt.Errorf("entrypoints[%d]: model name %q is already %s; entrypoint names must be new virtual names", index, name, meaning)
			}
		}

		result = append(result, EntrypointMapping{
			ModelNames: names,
			Recipe:     recipeName,
		})
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
			Name:        recipe.Name,
			Description: recipe.Description,
			Routing: CanonicalRouting{
				Signals:     canonicalSignalsFromSignals(recipe.Signals),
				Projections: canonicalProjectionsFromProjections(recipe.Projections),
				Decisions:   copyDecisions(recipe.Decisions),
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
			Recipe:     entrypoint.Recipe,
		})
	}
	return entrypoints
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
