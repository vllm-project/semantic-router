package config

import (
	"fmt"
	"net/url"
	"slices"
	"strings"
)

// RecipeName identifies an isolated routing namespace.
type RecipeName string

// DefaultRecipeName names the routing profile normalized from the top-level
// `routing:` block. Additional named profiles come from `recipes:`.
const DefaultRecipeName RecipeName = "default"

// RoutingStrategy controls how matching decisions are ordered within one
// routing profile.
type RoutingStrategy string

const (
	RoutingStrategyPriority   RoutingStrategy = "priority"
	RoutingStrategyConfidence RoutingStrategy = "confidence"
	routingNamespaceSeparator                 = "::"
)

// Validate rejects strategy values outside the public routing contract.
func (s RoutingStrategy) Validate() error {
	switch s {
	case "", RoutingStrategyPriority, RoutingStrategyConfidence:
		return nil
	default:
		return fmt.Errorf("routing.strategy must be %q or %q, got %q", RoutingStrategyPriority, RoutingStrategyConfidence, s)
	}
}

// RoutingProfile contains all state whose names and execution are isolated by
// a recipe. Shared provider bindings, model assets, and runtime services stay
// on RouterConfig.
type RoutingProfile struct {
	Signals     Signals
	Projections Projections
	Decisions   []Decision
	Strategy    RoutingStrategy
}

// RoutingRecipe gives an isolated routing profile a stable name and optional
// request-facing description.
type RoutingRecipe struct {
	Name        RecipeName
	Description string
	Profile     RoutingProfile
}

// EntrypointMapping binds request-facing virtual model names to a named
// recipe. The virtual names never reach a backend; they only select which
// routing profile evaluates the request.
//
// Exactly one of Recipe or Rules is set. Recipe is the legacy, unconditional
// form: every caller resolves to the same recipe. Rules is the conditional
// form: the recipe depends on request headers and path, evaluated by
// ResolveEntrypoint (entrypoint_resolver.go). A conditional entrypoint has
// no implicit fallback — an unmatched caller is denied, never routed to the
// default recipe or passthrough.
type EntrypointMapping struct {
	ModelNames []string
	Recipe     RecipeName
	Rules      []EntrypointRule
}

// RoutingDecisionRef identifies a decision inside its owning recipe. The
// object pointer is stable after startup because normalized configs are
// immutable.
type RoutingDecisionRef struct {
	Recipe   RecipeName
	Decision *Decision
}

// RoutingNamespaceKey returns a readable internal key for any recipe-local
// name. Public API fields continue to expose the local name alongside recipe.
func RoutingNamespaceKey(recipeName RecipeName, localName string) string {
	localName = strings.TrimSpace(localName)
	if localName == "" {
		return ""
	}
	scope := RoutingNamespaceScope(recipeName)
	if scope == "" {
		return localName
	}
	return scope + routingNamespaceSeparator + url.QueryEscape(localName)
}

// RoutingNamespaceScope returns the escaped storage component for a named
// recipe. The default recipe intentionally keeps existing unscoped keys.
func RoutingNamespaceScope(recipeName RecipeName) string {
	normalizedRecipe := RecipeName(strings.TrimSpace(string(recipeName)))
	if normalizedRecipe == "" {
		normalizedRecipe = DefaultRecipeName
	}
	if normalizedRecipe == DefaultRecipeName {
		return ""
	}
	return url.QueryEscape(string(normalizedRecipe))
}

// RoutingDecisionKey is the decision-specific spelling of RoutingNamespaceKey.
func RoutingDecisionKey(recipeName RecipeName, decisionName string) string {
	return RoutingNamespaceKey(recipeName, decisionName)
}

// RoutingDecisionRefs returns every decision with its owning recipe for
// startup inventory such as replay-recorder and selector initialization.
func (c *RouterConfig) RoutingDecisionRefs() []RoutingDecisionRef {
	if c == nil {
		return nil
	}
	if len(c.Recipes) == 0 {
		refs := make([]RoutingDecisionRef, 0, len(c.Decisions))
		for i := range c.Decisions {
			refs = append(refs, RoutingDecisionRef{Recipe: DefaultRecipeName, Decision: &c.Decisions[i]})
		}
		return refs
	}
	refs := make([]RoutingDecisionRef, 0)
	for i := range c.Recipes {
		recipe := &c.Recipes[i]
		for j := range recipe.Profile.Decisions {
			refs = append(refs, RoutingDecisionRef{Recipe: recipe.Name, Decision: &recipe.Profile.Decisions[j]})
		}
	}
	return refs
}

// RecipeByName returns the normalized recipe with the given name.
func (c *RouterConfig) RecipeByName(name RecipeName) (*RoutingRecipe, bool) {
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

// DefaultRecipe returns the recipe backing the flat routing fields. Canonical
// configs contain an explicit normalized default recipe; programmatically
// assembled single-profile configs get an equivalent immutable view so callers
// do not need a second routing path.
func (c *RouterConfig) DefaultRecipe() *RoutingRecipe {
	if c == nil {
		return nil
	}
	recipe, ok := c.RecipeByName(DefaultRecipeName)
	if ok {
		return recipe
	}
	return &RoutingRecipe{
		Name: DefaultRecipeName,
		Profile: RoutingProfile{
			Signals:     c.Signals,
			Projections: c.Projections,
			Decisions:   c.Decisions,
			Strategy:    c.Strategy,
		},
	}
}

// EntrypointByModelName returns the entrypoint claiming a request-facing
// virtual model name, if any. This is a pure alias-membership test: it
// returns ok=true for a conditional entrypoint's name even when no caller
// has been evaluated yet, so callers can distinguish "not an entrypoint at
// all" (ok=false) from "an entrypoint alias whose recipe depends on the
// caller" (ok=true, len(Rules) > 0).
func (c *RouterConfig) EntrypointByModelName(modelName string) (*EntrypointMapping, bool) {
	if c == nil {
		return nil, false
	}
	trimmed := strings.TrimSpace(modelName)
	if trimmed == "" {
		return nil, false
	}
	for i := range c.Entrypoints {
		if slices.Contains(c.Entrypoints[i].ModelNames, trimmed) {
			return &c.Entrypoints[i], true
		}
	}
	return nil, false
}

// RecipeForRequestModel resolves a request model name through the entrypoint
// table. It returns false when the name matches no entrypoint, or when the
// entrypoint is conditional (its recipe depends on request headers/path that
// this context-free signature has no way to receive — callers needing that
// must use ResolveEntrypoint instead). Callers fall back to auto-model or
// specified-model handling.
func (c *RouterConfig) RecipeForRequestModel(modelName string) (*RoutingRecipe, bool) {
	entrypoint, ok := c.EntrypointByModelName(modelName)
	if !ok || len(entrypoint.Rules) > 0 {
		return nil, false
	}
	return c.RecipeByName(entrypoint.Recipe)
}

// RecipeForRoutingModel resolves every request-facing routing model. Built-in
// auto and direct-looper aliases select the default recipe; named entrypoints
// select their mapped recipe. Concrete backend model IDs intentionally do not
// resolve to a recipe.
func (c *RouterConfig) RecipeForRoutingModel(modelName string) (*RoutingRecipe, bool) {
	if c == nil {
		return nil, false
	}
	trimmed := strings.TrimSpace(modelName)
	if c.IsAutoModelName(trimmed) || c.IsReMoMModelName(trimmed) || c.IsFusionModelName(trimmed) || c.IsFlowModelName(trimmed) {
		recipe := c.DefaultRecipe()
		return recipe, recipe != nil
	}
	return c.RecipeForRequestModel(modelName)
}

// ReachableRoutingRecipes returns the profiles that a request-facing routing
// model can select. The default profile is reachable through configured auto or
// direct-looper aliases; named profiles are reachable only through entrypoints.
// Startup resource discovery should use this view instead of treating every
// declared recipe as request reachable.
func (c *RouterConfig) ReachableRoutingRecipes() []*RoutingRecipe {
	if c == nil {
		return nil
	}

	reachable := make(map[RecipeName]struct{}, len(c.Entrypoints)+1)
	if c.defaultRecipeHasRoutingEntrypoint() {
		reachable[DefaultRecipeName] = struct{}{}
	}
	for _, entrypoint := range c.Entrypoints {
		if len(normalizeAutoModelNames(entrypoint.ModelNames)) == 0 {
			continue
		}
		for _, name := range entrypointRecipeNames(entrypoint) {
			reachable[name] = struct{}{}
		}
	}

	if len(c.Recipes) == 0 {
		if _, ok := reachable[DefaultRecipeName]; !ok {
			return nil
		}
		return []*RoutingRecipe{c.DefaultRecipe()}
	}

	recipes := make([]*RoutingRecipe, 0, len(reachable))
	for i := range c.Recipes {
		recipe := &c.Recipes[i]
		if _, ok := reachable[recipe.Name]; ok {
			recipes = append(recipes, recipe)
		}
	}
	return recipes
}

// IsRecipeReachableForRouting reports whether a normalized recipe can be
// selected by a request-facing model name.
func (c *RouterConfig) IsRecipeReachableForRouting(name RecipeName) bool {
	for _, recipe := range c.ReachableRoutingRecipes() {
		if recipe != nil && recipe.Name == name {
			return true
		}
	}
	return false
}

func (c *RouterConfig) defaultRecipeHasRoutingEntrypoint() bool {
	if len(c.EffectiveAutoModelNames()) > 0 {
		return true
	}
	if len(c.ExposedReMoMModelNames()) > 0 ||
		len(c.ExposedFusionModelNames()) > 0 ||
		len(c.ExposedFlowModelNames()) > 0 {
		return true
	}
	for _, entrypoint := range c.Entrypoints {
		if len(normalizeAutoModelNames(entrypoint.ModelNames)) == 0 {
			continue
		}
		for _, name := range entrypointRecipeNames(entrypoint) {
			if name == DefaultRecipeName {
				return true
			}
		}
	}
	return false
}

// ConfigForRecipe returns an immutable routing view over the shared router
// configuration. The returned value owns recipe-local routing fields while
// reusing read-only provider, model, and service configuration. Callers must
// not mutate either the returned config or the source config after startup.
func (c *RouterConfig) ConfigForRecipe(recipe *RoutingRecipe) *RouterConfig {
	if c == nil || recipe == nil {
		return nil
	}

	scoped := *c
	scoped.RoutingScope = recipe.Name
	scoped.IntelligentRouting = IntelligentRouting{
		Signals:         recipe.Profile.Signals,
		Projections:     recipe.Profile.Projections,
		Decisions:       recipe.Profile.Decisions,
		Strategy:        recipe.Profile.Strategy,
		ModelSelection:  c.ModelSelection,
		ReasoningConfig: c.ReasoningConfig,
	}
	scoped.KnowledgeBases = knowledgeBasesForRoutingProfile(c.KnowledgeBases, recipe.Profile)
	// A scoped config represents exactly one routing profile. Keeping the full
	// recipe list here would make helpers such as AllRoutingDecisions escape the
	// selected recipe again.
	scoped.Recipes = nil
	return &scoped
}

func knowledgeBasesForRoutingProfile(catalog []KnowledgeBaseConfig, profile RoutingProfile) []KnowledgeBaseConfig {
	referenced := make(map[string]struct{}, len(profile.Signals.KBRules))
	for _, rule := range profile.Signals.KBRules {
		if rule.KB != "" {
			referenced[rule.KB] = struct{}{}
		}
	}
	for _, score := range profile.Projections.Scores {
		for _, input := range score.Inputs {
			if strings.EqualFold(input.Type, ProjectionInputKBMetric) && input.KB != "" {
				referenced[input.KB] = struct{}{}
			}
		}
	}
	if len(referenced) == 0 {
		return nil
	}

	filtered := make([]KnowledgeBaseConfig, 0, len(referenced))
	for _, kb := range catalog {
		if _, ok := referenced[kb.Name]; ok {
			filtered = append(filtered, kb)
		}
	}
	return filtered
}

// IsEntrypointModelName reports whether the name is a request-facing virtual
// model name from the entrypoint table. Such names never reach a backend; the
// router resolves them like auto-model aliases. This is a pure alias-
// membership test — it returns true for a conditional entrypoint's name even
// when no rule has been evaluated yet, so a name that is merely claimed but
// currently unmatched for a given caller is still never treated as a
// concrete backend model.
func (c *RouterConfig) IsEntrypointModelName(modelName string) bool {
	_, ok := c.EntrypointByModelName(modelName)
	return ok
}

// EntrypointRecipeDescription returns the model-listing description for an
// entrypoint's recipe: the recipe's own description when set, otherwise a
// generic label naming the recipe.
func (c *RouterConfig) EntrypointRecipeDescription(recipeName RecipeName) string {
	if recipe, ok := c.RecipeByName(recipeName); ok && strings.TrimSpace(recipe.Description) != "" {
		return recipe.Description
	}
	return fmt.Sprintf("Entrypoint for the %s routing recipe", recipeName)
}

// AllRoutingDecisions returns the decisions of every routing profile for
// startup-only resource discovery and whole-config inventory. Request-time
// routing must use ConfigForRecipe instead.
func (c *RouterConfig) AllRoutingDecisions() []Decision {
	if c == nil {
		return nil
	}
	if len(c.Recipes) == 0 {
		return c.Decisions
	}
	if len(c.Recipes) == 1 {
		return c.Recipes[0].Profile.Decisions
	}
	all := make([]Decision, 0, 2*len(c.Decisions))
	for i := range c.Recipes {
		all = append(all, c.Recipes[i].Profile.Decisions...)
	}
	return all
}

// HasRoutingDecisions reports whether any routing profile declares decisions,
// without the per-request allocation of AllRoutingDecisions. The flat gate
// `len(c.Decisions) == 0` is wrong for recipes-only configs, where every
// decision lives in a non-default recipe.
func (c *RouterConfig) HasRoutingDecisions() bool {
	if c == nil {
		return false
	}
	if len(c.Decisions) > 0 {
		return true
	}
	for i := range c.Recipes {
		if len(c.Recipes[i].Profile.Decisions) > 0 {
			return true
		}
	}
	return false
}

// RoutingProfileSignals returns the default profile's signals for canonical
// export. The flat Signals field mirrors this value for single-profile readers.
func (c *RouterConfig) RoutingProfileSignals() Signals {
	if c == nil {
		return Signals{}
	}
	if recipe := c.DefaultRecipe(); recipe != nil {
		return recipe.Profile.Signals
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
		return recipe.Profile.Projections
	}
	return c.Projections
}
