package dsl

import (
	"fmt"
	"sort"
	"strings"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type recipeRoutingFragment struct {
	Routing config.CanonicalRouting `yaml:"routing"`
}

type recipeBundle struct {
	Version string              `yaml:"version,omitempty"`
	Recipes []recipeBundleEntry `yaml:"recipes"`
}

type recipeBundleEntry struct {
	Name        string                  `yaml:"name"`
	Description string                  `yaml:"description,omitempty"`
	Routing     config.CanonicalRouting `yaml:"routing"`
}

// parseRecipeAuthoringYAML accepts the two provider-neutral values owned by
// the DSL boundary. A routing fragment is one anonymous profile; a Recipe
// bundle preserves the human name and description for one or more reusable
// profiles. Neither shape can carry deployment or publication state.
func parseRecipeAuthoringYAML(data []byte) (*config.RouterConfig, error) {
	fields, err := recipeAuthoringFields(data)
	if err != nil {
		return nil, err
	}

	if containsField(fields, "routing") {
		if len(fields) != 1 {
			return nil, fmt.Errorf(
				"routing fragment must contain exactly one top-level routing field, got %v",
				fields,
			)
		}
		return parseRoutingFragmentYAML(data)
	}
	if containsField(fields, "recipes") {
		if len(fields) != 1 && (len(fields) != 2 || !containsField(fields, "version")) {
			return nil, fmt.Errorf(
				"recipe bundle may contain only version and recipes, got %v",
				fields,
			)
		}
		return parseRecipeBundleYAML(data)
	}
	return nil, fmt.Errorf(
		"recipe authoring YAML must contain exactly one top-level routing or recipes field, got %v",
		fields,
	)
}

func recipeAuthoringFields(data []byte) ([]string, error) {
	var raw map[string]interface{}
	if err := yaml.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("decode Recipe authoring YAML: %w", err)
	}
	fields := make([]string, 0, len(raw))
	for field := range raw {
		fields = append(fields, field)
	}
	sort.Strings(fields)
	return fields, nil
}

func containsField(fields []string, expected string) bool {
	for _, field := range fields {
		if field == expected {
			return true
		}
	}
	return false
}

func parseRoutingFragmentYAML(data []byte) (*config.RouterConfig, error) {
	compiled, err := config.ParseRoutingYAMLBytes(data)
	if err != nil {
		return nil, fmt.Errorf("compile routing fragment: %w", err)
	}
	return compiled, nil
}

// parseRecipeBundleYAML projects a provider-neutral v0.3 Recipe catalog into
// the runtime view consumed by the decompiler. The version marker is optional
// for an editor fragment and required by maintained catalog files; neither
// shape can carry deployment or publication state.
func parseRecipeBundleYAML(data []byte) (*config.RouterConfig, error) {
	var source recipeBundle
	if err := yaml.UnmarshalStrict(data, &source); err != nil {
		return nil, fmt.Errorf("decode Recipe bundle: %w", err)
	}
	if source.Version != "" && source.Version != "v0.3" {
		return nil, fmt.Errorf("recipe bundle version must be v0.3, got %q", source.Version)
	}
	if len(source.Recipes) == 0 {
		return nil, fmt.Errorf("recipe bundle must contain at least one Recipe")
	}

	seenNames := make(map[string]struct{}, len(source.Recipes))
	runtimeConfig := &config.RouterConfig{Recipes: make([]config.RoutingRecipe, 0, len(source.Recipes))}
	for index, recipe := range source.Recipes {
		name := strings.TrimSpace(recipe.Name)
		if name == "" || name != recipe.Name {
			return nil, fmt.Errorf(
				"recipes[%d].name must be non-empty without surrounding whitespace",
				index,
			)
		}
		if _, duplicate := seenNames[name]; duplicate {
			return nil, fmt.Errorf("recipe bundle repeats Recipe name %q", name)
		}
		seenNames[name] = struct{}{}
		if len(recipe.Routing.Decisions) == 0 {
			return nil, fmt.Errorf("recipes[%s].routing.decisions must not be empty", name)
		}

		validatedInput, err := yaml.Marshal(recipeRoutingFragment{Routing: recipe.Routing})
		if err != nil {
			return nil, fmt.Errorf("encode Recipe %q routing: %w", name, err)
		}
		compiled, err := config.ParseRoutingYAMLBytes(validatedInput)
		if err != nil {
			return nil, fmt.Errorf("compile Recipe %q: %w", name, err)
		}
		runtimeConfig.Recipes = append(runtimeConfig.Recipes, config.RoutingRecipe{
			Name:        config.RecipeName(name),
			Description: recipe.Description,
			Profile: config.RoutingProfile{
				Signals: compiled.Signals, Projections: compiled.Projections,
				Decisions: compiled.Decisions, Strategy: compiled.Strategy,
			},
		})
	}
	return runtimeConfig, nil
}
