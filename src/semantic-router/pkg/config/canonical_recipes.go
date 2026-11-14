package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

// CanonicalEntrypoint maps request-facing virtual model names to a named
// recipe in the public v0.5 contract.
type CanonicalEntrypoint struct {
	API        InferenceAPI `yaml:"api,omitempty" jsonschema:"enum=chat,enum=systemone"`
	ModelNames []string     `yaml:"model_names"`
	Recipe     string       `yaml:"recipe"`
	// ClientAuth stays out of the published schema until the Router enforces
	// it on every request path; until then normalization rejects it.
	ClientAuth *EntrypointClientAuth `yaml:"client_auth,omitempty" jsonschema:"-"`
	// Default marks the chat entrypoint of the default recipe, which serves
	// the configuration's default routing. Several recipes need exactly one.
	Default bool `yaml:"default,omitempty"`
}

// CanonicalRecipe is a named routing profile: its signals, projections and
// decisions, the order it tries them in, and the model bindings and plugins
// its decisions default to. The model catalog stays shared across recipes.
type CanonicalRecipe struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
	// ModelBindings override global.model_catalog.bindings for this recipe.
	ModelBindings map[string]ModelBinding `yaml:"model_bindings,omitempty"`
	// Plugins override global.plugins of the same type, field by field, and a
	// decision's plugin overrides them in turn.
	Plugins               []DecisionPlugin         `yaml:"plugins,omitempty"`
	Signals               CanonicalSignals         `yaml:"signals,omitempty"`
	Projections           CanonicalProjections     `yaml:"projections,omitempty"`
	Decisions             []Decision               `yaml:"decisions,omitempty"`
	Strategy              RoutingStrategy          `yaml:"strategy,omitempty"`
	Fallback              *fallback.FallbackPolicy `yaml:"fallback,omitempty" json:"fallback,omitempty"`
	CandidateRequirements *CandidateRequirements   `yaml:"candidate_requirements,omitempty"`
}

// recipePath is the document path of a recipe.
func recipePath(name string) string {
	return fmt.Sprintf("recipes[%s]", name)
}

// applyCanonicalRecipeState validates and normalizes `recipes` and
// `entrypoints` into RouterConfig: every recipe with the effective plugins of
// its decisions, and the default recipe mirrored in the flat routing fields.
// A configuration without recipes routes with an empty default recipe.
func applyCanonicalRecipeState(cfg *RouterConfig, canonical *CanonicalConfig) error {
	if err := validateCanonicalRecipes(canonical); err != nil {
		return err
	}
	defaultRecipe, err := canonicalDefaultRecipe(canonical)
	if err != nil {
		return err
	}
	if err = validatePluginLayer("global.plugins", cfg.GlobalPlugins); err != nil {
		return err
	}

	recipes := make([]RoutingRecipe, 0, len(canonical.Recipes))
	for _, recipe := range canonical.Recipes {
		recipes = append(recipes, routingRecipeFromCanonical(cfg.RoutingDefaults, recipe))
	}
	if len(recipes) == 0 {
		recipes = append(recipes, routingRecipeFromCanonical(cfg.RoutingDefaults, CanonicalRecipe{Name: string(defaultRecipe)}))
	}
	for i := range recipes {
		if err = layerRecipePlugins(cfg.GlobalPlugins, &recipes[i]); err != nil {
			return err
		}
	}
	cfg.Recipes = recipes
	cfg.defaultRecipe = defaultRecipe
	cfg.mirrorDefaultRecipe()

	entrypoints, err := normalizeCanonicalEntrypoints(cfg, canonical, recipes)
	if err != nil {
		return err
	}
	cfg.Entrypoints = entrypoints
	cfg.SemanticCache.InUse, _ = cfg.responseCacheDemand()
	cfg.Memory.InUse = cfg.UsesMemory()
	return nil
}

func routingRecipeFromCanonical(defaults RoutingDefaults, recipe CanonicalRecipe) RoutingRecipe {
	decisions := copyDecisions(recipe.Decisions)
	ensureModelRefDefaults(decisions)
	return RoutingRecipe{
		Name:        RecipeName(recipe.Name),
		Description: recipe.Description,
		Profile: RoutingProfile{
			ModelBindings:         cloneModelMap(recipe.ModelBindings),
			CandidateRequirements: recipe.CandidateRequirements.Clone(),
			Plugins:               copyDecisionPlugins(recipe.Plugins),
			Signals:               normalizeSignals(recipe.Signals),
			Projections:           normalizeProjections(recipe.Projections),
			Decisions:             decisions,
			Strategy:              defaults.resolveStrategy(recipe.Strategy),
			Fallback:              defaults.resolveFallback(recipe.Fallback),
		},
	}
}

// mirrorDefaultRecipe copies the default recipe's profile into the flat
// routing fields, which single-profile read sites use.
func (c *RouterConfig) mirrorDefaultRecipe() {
	profile := c.DefaultRecipe().Profile
	c.ModelBindings = cloneModelMap(profile.ModelBindings)
	c.CandidateRequirements = profile.CandidateRequirements.Clone()
	c.Plugins = copyDecisionPlugins(profile.Plugins)
	c.DefaultPlugins = copyDecisionPlugins(profile.DefaultPlugins)
	c.Signals = profile.Signals
	c.Projections = profile.Projections
	c.Decisions = profile.Decisions
	c.Strategy = profile.Strategy
	c.Fallback = profile.Fallback.Clone()
}

func validateCanonicalRecipes(canonical *CanonicalConfig) error {
	cards, err := canonicalModelCardIndex(canonical.Providers.Models)
	if err != nil {
		return err
	}
	aliases := make(map[string]CanonicalProviderModel, len(canonical.Providers.Models))
	for _, model := range canonical.Providers.Models {
		aliases[model.Name] = model
	}

	seen := make(map[RecipeName]struct{}, len(canonical.Recipes))
	for _, recipe := range canonical.Recipes {
		name := RecipeName(strings.TrimSpace(recipe.Name))
		if name == "" {
			return fmt.Errorf("recipes[].name cannot be empty")
		}
		path := recipePath(string(name))
		if name == GlobalModelScope {
			return fmt.Errorf("%s: name is reserved for shared model services", path)
		}
		if string(name) != recipe.Name {
			return fmt.Errorf("%s.name must not contain surrounding whitespace", path)
		}
		if _, exists := seen[name]; exists {
			return fmt.Errorf("%s: duplicate recipe name", path)
		}
		seen[name] = struct{}{}

		if err := recipe.CandidateRequirements.Validate(); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if err := validatePluginLayer(path+".plugins", recipe.Plugins); err != nil {
			return err
		}
		if err := validateCanonicalDecisions(path, recipe.Decisions, aliases, cards); err != nil {
			return err
		}
	}
	return nil
}

// canonicalDefaultRecipe names the default recipe: the only recipe, or the
// recipe of the one entrypoint marked default when there are several.
func canonicalDefaultRecipe(canonical *CanonicalConfig) (RecipeName, error) {
	marked := -1
	for index, entrypoint := range canonical.Entrypoints {
		if !entrypoint.Default {
			continue
		}
		if entrypoint.API != "" && entrypoint.API != ChatAPI {
			return "", fmt.Errorf("entrypoints[%d].default: only a chat entrypoint can be the default", index)
		}
		if marked >= 0 {
			return "", fmt.Errorf("entrypoints[%d].default: entrypoints[%d] is already the default; mark exactly one entrypoint default", index, marked)
		}
		marked = index
	}
	switch {
	case len(canonical.Recipes) == 0 && len(canonical.Entrypoints) > 0:
		return "", fmt.Errorf("entrypoints: no recipe is declared for them to map")
	case len(canonical.Recipes) == 0:
		return DefaultRecipeName, nil
	case len(canonical.Recipes) == 1:
		return RecipeName(canonical.Recipes[0].Name), nil
	case len(canonical.Entrypoints) == 0:
		return "", fmt.Errorf("entrypoints: %d recipes need entrypoints, exactly one of them marked default: true", len(canonical.Recipes))
	case marked < 0:
		return "", fmt.Errorf("entrypoints: %d recipes need exactly one entrypoint marked default: true", len(canonical.Recipes))
	}
	return RecipeName(strings.TrimSpace(canonical.Entrypoints[marked].Recipe)), nil
}

func normalizeCanonicalEntrypoints(cfg *RouterConfig, canonical *CanonicalConfig, recipes []RoutingRecipe) ([]EntrypointMapping, error) {
	entrypoints := canonical.Entrypoints
	result := make([]EntrypointMapping, 0, len(entrypoints))
	claimed := make(map[InferenceAPI]map[string]struct{})
	for index, entrypoint := range entrypoints {
		api := entrypoint.API
		if api == "" {
			api = ChatAPI
		}
		if api != ChatAPI && api != SystemOneAPI {
			return nil, fmt.Errorf("entrypoints[%d].api must be chat or systemone", index)
		}
		if claimed[api] == nil {
			claimed[api] = map[string]struct{}{}
		}
		recipeName := RecipeName(strings.TrimSpace(entrypoint.Recipe))
		if recipeName == "" {
			return nil, fmt.Errorf("entrypoints[%d].recipe cannot be empty", index)
		}
		if findRecipe(recipes, recipeName) == nil {
			return nil, fmt.Errorf("entrypoints[%d]: unknown recipe %q", index, recipeName)
		}

		names := normalizeEntrypointNames(entrypoint.ModelNames)
		if len(names) == 0 {
			return nil, fmt.Errorf("entrypoints[%d].model_names cannot be empty", index)
		}
		for _, name := range names {
			if _, exists := claimed[api][name]; exists {
				return nil, fmt.Errorf("entrypoints[%d]: model name %q is already mapped by another entrypoint", index, name)
			}
			claimed[api][name] = struct{}{}
			if meaning := entrypointNameConflict(cfg, canonical, api, name); meaning != "" {
				return nil, fmt.Errorf("entrypoints[%d]: model name %q is already %s; entrypoint names must be new virtual names", index, name, meaning)
			}
		}
		clientAuth, err := normalizeEntrypointClientAuth(index, entrypoint.ClientAuth)
		if err != nil {
			return nil, err
		}

		result = append(result, EntrypointMapping{
			API:        entrypoint.API,
			ModelNames: names,
			Recipe:     recipeName,
			ClientAuth: clientAuth,
			Default:    entrypoint.Default,
		})
	}
	view := *cfg
	view.Entrypoints = result
	for _, entrypoint := range view.EffectiveEntrypoints(ChatAPI) {
		if entrypoint.Source != EntrypointBuiltin {
			continue
		}
		for _, name := range entrypoint.ModelNames {
			if meaning := configuredEntrypointNameConflict(canonical, name); meaning != "" {
				return nil, fmt.Errorf("default entrypoint %q is already %s; declare an entrypoint with default: true and other model_names", name, meaning)
			}
		}
	}
	return result, nil
}

// entrypointNameConflict reports what an entrypoint virtual name already means
// to the router, if anything. Entrypoint names must be new: reusing an existing
// routable name would silently hijack it, because requestModelActsAsAuto stops
// treating the name as an explicitly specified model.
func entrypointNameConflict(cfg *RouterConfig, canonical *CanonicalConfig, api InferenceAPI, name string) string {
	if api == SystemOneAPI {
		for _, deployment := range cfg.ModelDeployments {
			if deployment.PublicModelName() == name {
				return "a concrete System One model"
			}
		}
		for _, model := range canonical.Providers.Models {
			if model.APIFormat == APIFormatSystemOne && model.Name == name {
				return "a native provider model"
			}
		}
		return ""
	}
	return configuredEntrypointNameConflict(canonical, name)
}

func configuredEntrypointNameConflict(canonical *CanonicalConfig, name string) string {
	for _, model := range canonical.Providers.Models {
		if model.APIFormat == APIFormatSystemOne {
			continue
		}
		if model.Name == name || model.ProviderModelID == name {
			return "a configured model"
		}
		for _, externalID := range model.ExternalModelIDs {
			if externalID == name {
				return "a configured provider model alias"
			}
		}
	}
	for _, card := range canonicalModelCards(canonical.Providers.Models) {
		if routingModelHasLoRA(card, name) {
			return "a configured LoRA adapter"
		}
	}
	return ""
}

// canonicalRecipesFromRouterConfig exports every normalized recipe, with the
// plugins its decisions declare.
func canonicalRecipesFromRouterConfig(cfg *RouterConfig) []CanonicalRecipe {
	if cfg == nil {
		return nil
	}
	if len(cfg.Recipes) == 0 {
		return []CanonicalRecipe{CanonicalRecipeFromRouting(*cfg.DefaultRecipe())}
	}
	recipes := make([]CanonicalRecipe, 0, len(cfg.Recipes))
	for _, recipe := range cfg.Recipes {
		recipes = append(recipes, CanonicalRecipeFromRouting(recipe))
	}
	return recipes
}

// CanonicalRecipeFromRouting exports one normalized recipe as it is written.
func CanonicalRecipeFromRouting(recipe RoutingRecipe) CanonicalRecipe {
	decisions := copyDecisions(recipe.Profile.Decisions)
	for i := range decisions {
		decisions[i].Plugins = decisions[i].DeclaredPlugins()
		decisions[i].declaredPlugins, decisions[i].pluginsLayered = nil, false
	}
	return CanonicalRecipe{
		Name:                  string(recipe.Name),
		Description:           recipe.Description,
		ModelBindings:         cloneModelMap(recipe.Profile.ModelBindings),
		Plugins:               copyDecisionPlugins(recipe.Profile.Plugins),
		Signals:               canonicalSignalsFromSignals(recipe.Profile.Signals),
		Projections:           canonicalProjectionsFromProjections(recipe.Profile.Projections),
		Decisions:             decisions,
		Strategy:              recipe.Profile.Strategy,
		Fallback:              recipe.Profile.Fallback.Clone(),
		CandidateRequirements: recipe.Profile.CandidateRequirements.Clone(),
	}
}

// canonicalEntrypointsFromRouterConfig exports the normalized entrypoint table.
func canonicalEntrypointsFromRouterConfig(cfg *RouterConfig) []CanonicalEntrypoint {
	if cfg == nil || len(cfg.Entrypoints) == 0 {
		return nil
	}
	entrypoints := make([]CanonicalEntrypoint, 0, len(cfg.Entrypoints))
	for _, entrypoint := range cfg.Entrypoints {
		entrypoints = append(entrypoints, CanonicalEntrypoint{
			API:        entrypoint.API,
			ModelNames: append([]string(nil), entrypoint.ModelNames...),
			Recipe:     string(entrypoint.Recipe),
			ClientAuth: entrypoint.ClientAuth.Clone(),
			Default:    entrypoint.Default,
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
