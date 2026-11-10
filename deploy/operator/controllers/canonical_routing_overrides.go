package controllers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"

	"gopkg.in/yaml.v3"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// decodeCanonicalList decodes a pass-through list of canonical objects, and
// for each the keys it sets.
func decodeCanonicalList[T any](raw *apiextensionsv1.JSON) ([]T, []map[string]bool, error) {
	if raw == nil || len(raw.Raw) == 0 {
		return nil, nil, nil
	}
	var objects []map[string]interface{}
	if err := json.Unmarshal(raw.Raw, &objects); err != nil {
		return nil, nil, fmt.Errorf("must be a list of objects: %w", err)
	}
	data, err := yaml.Marshal(objects)
	if err != nil {
		return nil, nil, err
	}
	var items []T
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&items); err != nil {
		return nil, nil, err
	}
	keys := make([]map[string]bool, len(objects))
	for index, object := range objects {
		keys[index] = make(map[string]bool, len(object))
		for key := range object {
			keys[index][key] = true
		}
	}
	return items, keys, nil
}

// applyOperatorRecipes puts config.recipes, config.entrypoints, config.plugins
// and config.models into the canonical config.
func applyOperatorRecipes(canonical *routerconfig.CanonicalConfig, spec configRoutingSpec) error {
	recipes, keys, err := decodeCanonicalList[routerconfig.CanonicalRecipe](spec.recipes)
	if err != nil {
		return fmt.Errorf("config.recipes: %w", err)
	}
	for index, recipe := range recipes {
		if validateErr := recipe.CandidateRequirements.Validate(); validateErr != nil {
			return fmt.Errorf("config.recipes[%d].candidate_requirements: %w", index, validateErr)
		}
		if recipe.Name == string(routerconfig.DefaultRecipeName) {
			overrideRecipe(defaultRecipe(canonical), recipe, keys[index])
			continue
		}
		canonical.Recipes = append(canonical.Recipes, recipe)
	}
	entrypoints, _, err := decodeCanonicalList[routerconfig.CanonicalEntrypoint](spec.entrypoints)
	if err != nil {
		return fmt.Errorf("config.entrypoints: %w", err)
	}
	if spec.entrypoints != nil {
		canonical.Entrypoints = entrypoints
	}
	plugins, _, err := decodeCanonicalList[routerconfig.DecisionPlugin](spec.plugins)
	if err != nil {
		return fmt.Errorf("config.plugins: %w", err)
	}
	if spec.plugins != nil {
		canonical.Global.Plugins = plugins
	}
	models, _, err := decodeCanonicalList[routerconfig.CanonicalProviderModel](spec.models)
	if err != nil {
		return fmt.Errorf("config.models: %w", err)
	}
	return mergeModelFacts(canonical, models)
}

type configRoutingSpec struct {
	recipes, entrypoints, plugins, models *apiextensionsv1.JSON
}

// defaultRecipe is the operator's default recipe, which it always writes.
func defaultRecipe(canonical *routerconfig.CanonicalConfig) *routerconfig.CanonicalRecipe {
	for index := range canonical.Recipes {
		if canonical.Recipes[index].Name == string(routerconfig.DefaultRecipeName) {
			return &canonical.Recipes[index]
		}
	}
	canonical.Recipes = append([]routerconfig.CanonicalRecipe{{Name: string(routerconfig.DefaultRecipeName)}}, canonical.Recipes...)
	return &canonical.Recipes[0]
}

// overrideRecipe sets the fields of target that recipe sets.
func overrideRecipe(target *routerconfig.CanonicalRecipe, recipe routerconfig.CanonicalRecipe, keys map[string]bool) {
	if keys["description"] {
		target.Description = recipe.Description
	}
	if keys["model_bindings"] {
		target.ModelBindings = recipe.ModelBindings
	}
	if keys["plugins"] {
		target.Plugins = recipe.Plugins
	}
	if keys["signals"] {
		target.Signals = recipe.Signals
	}
	if keys["projections"] {
		target.Projections = recipe.Projections
	}
	if keys["decisions"] {
		target.Decisions = recipe.Decisions
	}
	if keys["strategy"] {
		target.Strategy = recipe.Strategy
	}
	if keys["fallback"] {
		target.Fallback = recipe.Fallback
	}
	if keys["candidate_requirements"] {
		target.CandidateRequirements = recipe.CandidateRequirements
	}
}

// mergeModelFacts states config.models' facts on the providers.models[]
// entries of the same name, and adds an entry for a name none has.
func mergeModelFacts(canonical *routerconfig.CanonicalConfig, models []routerconfig.CanonicalProviderModel) error {
	for _, model := range models {
		transport := model
		transport.ModelFacts = routerconfig.ModelFacts{}
		if model.Name == "" || !reflect.DeepEqual(transport, routerconfig.CanonicalProviderModel{Name: model.Name}) {
			return fmt.Errorf("config.models[%s]: an entry states a name and facts; its transport comes from vllmEndpoints", model.Name)
		}
		found := false
		for index := range canonical.Providers.Models {
			if canonical.Providers.Models[index].Name == model.Name {
				canonical.Providers.Models[index].ModelFacts = model.ModelFacts
				found = true
			}
		}
		if !found {
			canonical.Providers.Models = append(canonical.Providers.Models, model)
		}
	}
	return nil
}
