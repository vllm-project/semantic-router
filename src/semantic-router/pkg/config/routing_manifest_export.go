package config

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// CanonicalRoutingManifestFromSnapshot renders only the portable v0.3 routing
// authority. Runtime, identity, access, generated IDs, revisions, and secret
// values are deliberately absent.
func CanonicalRoutingManifestFromSnapshot(snapshot *routingsnapshot.Snapshot) (CanonicalConfig, error) {
	if snapshot == nil {
		return CanonicalConfig{}, fmt.Errorf("routing snapshot is required")
	}
	manifest := CanonicalConfig{Version: "v0.3"}
	if snapshot.Currency != "" {
		manifest.Global = &CanonicalGlobal{Billing: &CanonicalBillingGlobal{Currency: snapshot.Currency}}
	}
	modelsByID, err := appendCanonicalManifestModels(&manifest, snapshot.Models)
	if err != nil {
		return CanonicalConfig{}, err
	}
	recipesByID, err := appendCanonicalManifestRecipes(&manifest, snapshot.Recipes)
	if err != nil {
		return CanonicalConfig{}, err
	}
	if err := appendCanonicalManifestEntrypoints(&manifest, snapshot, modelsByID, recipesByID); err != nil {
		return CanonicalConfig{}, err
	}
	sortCanonicalRoutingManifest(&manifest)
	return manifest, nil
}

func appendCanonicalManifestModels(
	manifest *CanonicalConfig,
	models []routingsnapshot.Model,
) (map[string]string, error) {
	modelsByID := make(map[string]string, len(models))
	for _, model := range models {
		modelsByID[model.ID] = model.Name
		provider, err := canonicalProviderModelFromSnapshot(model)
		if err != nil {
			return nil, err
		}
		manifest.Providers.Models = append(manifest.Providers.Models, provider)
		manifest.Routing.ModelCards = append(manifest.Routing.ModelCards, canonicalModelCardFromSnapshot(model))
	}
	return modelsByID, nil
}

func canonicalProviderModelFromSnapshot(model routingsnapshot.Model) (CanonicalProviderModel, error) {
	provider := CanonicalProviderModel{
		Name: model.Name, ProviderModelID: model.Name,
		Control: modelControlFromExecution(model.Execution), Pricing: modelPricingFromSnapshot(model.Pricing),
	}
	for _, backend := range model.Backends {
		weight, err := portableBackendWeight(backend.Weight)
		if err != nil {
			return CanonicalProviderModel{}, fmt.Errorf("export Model %q: %w", model.Name, err)
		}
		if backend.ProviderModelID != provider.ProviderModelID && provider.ProviderModelID != model.Name {
			return CanonicalProviderModel{}, fmt.Errorf("model %q has backend-specific provider Model IDs that portable v0.3 cannot represent", model.Name)
		}
		provider.ProviderModelID = backend.ProviderModelID
		provider.BackendRefs = append(provider.BackendRefs, CanonicalBackendRef{
			Provider: backend.ProviderID, Type: string(backend.WireFormat),
			Endpoint:   backend.Origin,
			Credential: backend.ProviderCredentialID, Weight: weight,
			ChatPath: backend.Connection.Path, ExtraHeaders: clonePublicBackendHeaders(backend.Connection.Headers),
		})
	}
	if provider.ProviderModelID == model.Name {
		provider.ProviderModelID = ""
	}
	return provider, nil
}

func canonicalModelCardFromSnapshot(model routingsnapshot.Model) RoutingModel {
	loras := make([]LoRAAdapter, 0, len(model.LoRAs))
	for _, name := range model.LoRAs {
		loras = append(loras, LoRAAdapter{Name: name})
	}
	return RoutingModel{
		Name: model.Name, ParamSize: model.ParamSize, ContextWindowSize: model.ContextWindowSize,
		Description: model.Description, Capabilities: append([]string(nil), model.Capabilities...),
		Reasoning: ModelReasoning{Type: model.Reasoning.Type, Efforts: append([]string(nil), model.Reasoning.Efforts...)},
		LoRAs:     loras, QualityScore: model.QualityScore, Modality: model.Modality,
		Tags: append([]string(nil), model.Tags...),
	}
}

func appendCanonicalManifestRecipes(
	manifest *CanonicalConfig,
	recipes []routingsnapshot.Recipe,
) (map[string]string, error) {
	recipesByID := make(map[string]string, len(recipes))
	for _, recipe := range recipes {
		document, _, err := ParseRoutingRecipeDocument(recipe.Document)
		if err != nil {
			return nil, fmt.Errorf("export Recipe %q: %w", recipe.Name, err)
		}
		recipesByID[recipe.ID] = recipe.Name
		manifest.Recipes = append(manifest.Recipes, CanonicalRecipe{
			Name: recipe.Name, Description: recipe.Description, Routing: CanonicalRoutingFromRecipeDocument(document),
		})
	}
	return recipesByID, nil
}

func appendCanonicalManifestEntrypoints(
	manifest *CanonicalConfig,
	snapshot *routingsnapshot.Snapshot,
	modelsByID, recipesByID map[string]string,
) error {
	for _, entrypoint := range snapshot.Entrypoints {
		if len(entrypoint.Rules) != 1 || len(entrypoint.Rules[0].Matchers) != 0 {
			return fmt.Errorf("entrypoint %q cannot be represented by portable v0.3", entrypoint.Name)
		}
		rule := entrypoint.Rules[0]
		recipeName, found := recipesByID[rule.RecipeID]
		if !found {
			return fmt.Errorf("entrypoint %q references an absent Recipe", entrypoint.Name)
		}
		assignments, err := canonicalEntrypointAssignments(entrypoint.Name, rule, snapshot.Recipes, modelsByID)
		if err != nil {
			return err
		}
		modelNames := stableUniqueStrings(append([]string{entrypoint.Name}, entrypoint.Aliases...))
		manifest.Entrypoints = append(manifest.Entrypoints, CanonicalEntrypoint{
			ModelNames: modelNames, Recipe: recipeName, Assignments: assignments,
		})
	}
	return nil
}

func canonicalEntrypointAssignments(
	entrypointName string,
	rule routingsnapshot.EntrypointRule,
	recipes []routingsnapshot.Recipe,
	modelsByID map[string]string,
) (map[string]EntrypointAssignmentSet, error) {
	assignments := make(map[string]EntrypointAssignmentSet, len(rule.Assignments))
	for _, recipe := range recipes {
		if recipe.ID != rule.RecipeID {
			continue
		}
		decisionNames := make(map[string]string, len(recipe.Decisions))
		for _, decision := range recipe.Decisions {
			decisionNames[decision.ID] = decision.Name
		}
		for decisionID, set := range rule.Assignments {
			name, ok := decisionNames[decisionID]
			if !ok {
				return nil, fmt.Errorf("entrypoint %q references an absent Decision", entrypointName)
			}
			authoring, err := canonicalAuthoringAssignmentSet(entrypointName, set, modelsByID)
			if err != nil {
				return nil, err
			}
			assignments[name] = authoringAssignmentSetToPublic(authoring)
		}
	}
	return assignments, nil
}

func canonicalAuthoringAssignmentSet(
	entrypointName string,
	set routingsnapshot.AssignmentSet,
	modelsByID map[string]string,
) (AuthoringAssignmentSet, error) {
	authoring := AuthoringAssignmentSet{Fallback: nil}
	if set.Fallback != nil {
		authoring.Fallback = &AuthoringFallbackPolicy{
			Strategy: set.Fallback.Strategy,
			On:       append([]string(nil), set.Fallback.On...),
		}
	}
	for _, assignment := range set.Models {
		modelName, exists := modelsByID[assignment.ModelID]
		if !exists {
			return AuthoringAssignmentSet{}, fmt.Errorf("entrypoint %q references an absent Model", entrypointName)
		}
		var reasoning *AuthoringAssignmentReasoning
		if assignment.Reasoning != nil {
			reasoning = &AuthoringAssignmentReasoning{
				Enabled: assignment.Reasoning.Enabled, Effort: assignment.Reasoning.Effort,
				Description: assignment.Reasoning.Description,
			}
		}
		authoring.Models = append(authoring.Models, AuthoringModelAssignment{
			Model: modelName, Priority: assignment.Priority, Weight: assignment.Weight,
			LoRAName: assignment.LoRAName, Reasoning: reasoning,
		})
	}
	return authoring, nil
}

func sortCanonicalRoutingManifest(manifest *CanonicalConfig) {
	sort.Slice(manifest.Providers.Models, func(i, j int) bool {
		return manifest.Providers.Models[i].Name < manifest.Providers.Models[j].Name
	})
	sort.Slice(manifest.Routing.ModelCards, func(i, j int) bool {
		return manifest.Routing.ModelCards[i].Name < manifest.Routing.ModelCards[j].Name
	})
	sort.Slice(manifest.Recipes, func(i, j int) bool {
		return manifest.Recipes[i].Name < manifest.Recipes[j].Name
	})
	sort.Slice(manifest.Entrypoints, func(i, j int) bool {
		return manifest.Entrypoints[i].ModelNames[0] < manifest.Entrypoints[j].ModelNames[0]
	})
}

func modelControlFromExecution(value routingsnapshot.ModelExecution) ModelControl {
	var control ModelControl
	if value.MaxRetries != 0 || len(value.RetryOn) != 0 {
		control.Retry = &ModelRetry{Count: value.MaxRetries, On: append([]string(nil), value.RetryOn...)}
	}
	if value.RequestTimeout != "" || value.StreamTimeout != "" {
		control.Timeout = &ModelTimeout{Request: value.RequestTimeout, Stream: value.StreamTimeout}
	}
	return control
}

func modelPricingFromSnapshot(value routingsnapshot.ModelPricing) ModelRuntimePricing {
	return ModelRuntimePricing{
		InputCostPerMillionTokens:      cloneStringPointer(value.InputCostPerMillionTokens),
		OutputCostPerMillionTokens:     cloneStringPointer(value.OutputCostPerMillionTokens),
		CacheReadCostPerMillionTokens:  cloneStringPointer(value.CacheReadCostPerMillionTokens),
		CacheWriteCostPerMillionTokens: cloneStringPointer(value.CacheWriteCostPerMillionTokens),
	}
}

func portableBackendWeight(value string) (int, error) {
	if value == "" {
		return 0, nil
	}
	weight, err := strconv.Atoi(value)
	if err != nil || weight < 0 {
		return 0, fmt.Errorf("backend weight %q is not representable by strict v0.3", value)
	}
	return weight, nil
}
