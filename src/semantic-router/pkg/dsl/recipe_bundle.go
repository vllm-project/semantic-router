package dsl

import (
	"fmt"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// parseRecipeBundleYAML projects a provider-neutral v0.4 Recipe bundle into
// the runtime view consumed by the decompiler. Recipe bundles deliberately
// cannot carry Models, Entrypoints, listeners, billing, or global settings.
func parseRecipeBundleYAML(data []byte) (*config.RouterConfig, error) {
	var source config.CanonicalConfig
	if err := yaml.UnmarshalStrict(data, &source); err != nil {
		return nil, fmt.Errorf("decode Recipe bundle: %w", err)
	}
	if source.Version != "v0.4" || len(source.Recipes) == 0 || len(source.Listeners) != 0 ||
		len(source.Models) != 0 || len(source.Entrypoints) != 0 || source.Global != nil {
		return nil, fmt.Errorf("recipe bundle must contain only version v0.4 and recipes")
	}
	if _, err := config.CompileStandaloneRoutingSnapshot(source, nil); err != nil {
		return nil, fmt.Errorf("compile Recipe bundle: %w", err)
	}

	runtimeConfig := &config.RouterConfig{Recipes: make([]config.RoutingRecipe, 0, len(source.Recipes))}
	for _, recipe := range source.Recipes {
		document, err := yaml.Marshal(struct {
			Document config.CanonicalRouting `yaml:"document"`
		}{Document: recipe.Document})
		if err != nil {
			return nil, fmt.Errorf("encode Recipe %q: %w", recipe.Name, err)
		}
		compiled, err := config.ParseRoutingYAMLBytes(document)
		if err != nil {
			return nil, fmt.Errorf("compile Recipe %q: %w", recipe.Name, err)
		}
		runtimeConfig.Recipes = append(runtimeConfig.Recipes, config.RoutingRecipe{
			Name:        config.RecipeName(recipe.Name),
			Description: recipe.Description,
			Profile: config.RoutingProfile{
				Signals: compiled.Signals, Projections: compiled.Projections,
				Decisions: compiled.Decisions, Strategy: compiled.Strategy,
			},
		})
	}
	return runtimeConfig, nil
}
