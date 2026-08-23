package config

import (
	"fmt"
	"net/url"
	"slices"
	"strings"
)

// RecipeName identifies an isolated routing namespace.
type RecipeName string

// DefaultRecipeName identifies the optional explicit Recipe that backs
// model-less classification APIs. Request routing still requires an Entrypoint.
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
	ID           string
	Revision     int64
	Name         RecipeName
	Description  string
	Profile      RoutingProfile
	runtimeScope RecipeName
}

// RuntimeScope returns the stable namespace used by mutable routing state.
// Reusable base recipes use their canonical name; entrypoint-derived views use
// a deterministic identity for the whole entrypoint mapping. The logical Name
// remains unchanged for canonical/API recipe identity.
func (r *RoutingRecipe) RuntimeScope() RecipeName {
	if r == nil {
		return ""
	}
	if r.runtimeScope != "" {
		return r.runtimeScope
	}
	return r.Name
}

// EntrypointMapping is the runtime form of one v0.4 callable virtual Model.
// Each rule owns a complete action; no detached model-binding state exists.
type EntrypointMapping struct {
	ID         string
	Revision   int64
	Name       string
	ModelNames []string
	Rules      []EntrypointRule

	// Recipe and derivedRecipe identify the unconditional rule used by the
	// request-facing virtual Model. They are compiled from Rules and are never
	// accepted as another public authoring shape.
	Recipe RecipeName

	derivedRecipe *RoutingRecipe
}

type EntrypointRule struct {
	ID      string
	Name    string
	Matches []EntrypointMatch
	Action  EntrypointRuleAction

	derivedRecipe *RoutingRecipe
}

type EntrypointMatch struct {
	Claim *EntrypointClaimMatch
	Path  *EntrypointPathMatch
}

type EntrypointClaimMatch struct {
	Name  string
	Value EntrypointClaimValue
}

type EntrypointClaimValue struct {
	Kind    string
	String  string
	Boolean bool
	Integer int64
}

type EntrypointPathMatch struct {
	Exact  string
	Prefix string
}

type EntrypointRuleAction struct {
	RecipeID       string
	RecipeRevision int64
	Recipe         RecipeName
	Assignments    map[string]RoutingAssignmentSet
}

type RoutingAssignmentSet struct {
	Models   []RoutingModelAssignment
	Fallback *RoutingFallbackPolicy
}

type RoutingFallbackPolicy struct {
	Strategy string
	On       []string
}

type RoutingModelAssignment struct {
	ModelID       string
	ModelRevision int64
	ModelName     string
	Priority      int
	Weight        string
	LoRAName      string
	Reasoning     *RoutingAssignmentReasoning
}

type RoutingAssignmentReasoning struct {
	Enabled     bool
	Effort      string
	Description string
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
	refs := make([]RoutingDecisionRef, 0)
	if c.RoutingScope != "" {
		for i := range c.Decisions {
			refs = append(refs, RoutingDecisionRef{Recipe: c.RoutingScope, Decision: &c.Decisions[i]})
		}
		return refs
	}
	for i := range c.Recipes {
		recipe := &c.Recipes[i]
		for j := range recipe.Profile.Decisions {
			refs = append(refs, RoutingDecisionRef{Recipe: recipe.Name, Decision: &recipe.Profile.Decisions[j]})
		}
	}
	for i := range c.Entrypoints {
		for ruleIndex := range c.Entrypoints[i].Rules {
			recipe := c.Entrypoints[i].Rules[ruleIndex].derivedRecipe
			if recipe == nil || recipe.RuntimeScope() == recipe.Name {
				continue
			}
			for j := range recipe.Profile.Decisions {
				refs = append(refs, RoutingDecisionRef{Recipe: recipe.RuntimeScope(), Decision: &recipe.Profile.Decisions[j]})
			}
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

// RecipeByRuntimeScope resolves a stable internal routing scope back to the
// immutable effective recipe view that owns it. This is used to restore exact
// entrypoint bindings on router-generated looper requests.
func (c *RouterConfig) RecipeByRuntimeScope(scope RecipeName) (*RoutingRecipe, bool) {
	if c == nil || scope == "" {
		return nil, false
	}
	for i := range c.Entrypoints {
		for ruleIndex := range c.Entrypoints[i].Rules {
			recipe := c.Entrypoints[i].Rules[ruleIndex].derivedRecipe
			if recipe != nil && recipe.RuntimeScope() == scope {
				return recipe, true
			}
		}
	}
	return c.RecipeByName(scope)
}

// DefaultRecipe returns the explicit Recipe named "default". Runtime config
// never synthesizes a Recipe from the scoped flat routing view.
func (c *RouterConfig) DefaultRecipe() *RoutingRecipe {
	if c == nil {
		return nil
	}
	recipe, ok := c.RecipeByName(DefaultRecipeName)
	if !ok {
		return nil
	}
	return recipe
}

// RecipeForRequestModel resolves a request model name through the entrypoint
// table. It returns false when the name matches no Entrypoint.
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
			// Model-only resolution has no trusted path or claim context, so
			// it may select only the Entrypoint's explicit unconditional rule.
			return entrypoint.derivedRecipe, entrypoint.derivedRecipe != nil
		}
	}
	return nil, false
}

// ReachableRoutingRecipes returns the profiles that a request-facing routing
// Entrypoint can select. Startup resource discovery should use this view
// instead of treating every declared Recipe as request reachable. Bound
// Entrypoints contribute their derived views rather than the reusable base
// Recipe, because their effective model targets can differ even when several
// Entrypoints name the same Recipe.
func (c *RouterConfig) ReachableRoutingRecipes() []*RoutingRecipe {
	if c == nil {
		return nil
	}
	return c.entrypointRecipes()
}

func (c *RouterConfig) entrypointRecipes() []*RoutingRecipe {
	recipes := make([]*RoutingRecipe, 0, len(c.Entrypoints))
	for i := range c.Entrypoints {
		entrypoint := &c.Entrypoints[i]
		if len(entrypoint.ModelNames) == 0 {
			continue
		}
		for ruleIndex := range entrypoint.Rules {
			if derived := entrypoint.Rules[ruleIndex].derivedRecipe; derived != nil {
				recipes = append(recipes, derived)
			}
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

// ConfigForRecipe returns an immutable routing view over the shared router
// configuration. The returned value owns recipe-local routing fields while
// reusing read-only provider, model, and service configuration. Callers must
// not mutate either the returned config or the source config after startup.
func (c *RouterConfig) ConfigForRecipe(recipe *RoutingRecipe) *RouterConfig {
	if c == nil || recipe == nil {
		return nil
	}

	scoped := *c
	// RoutingScope remains the logical recipe identity because classifiers use
	// it in public metric labels. Mutable per-entrypoint state is keyed from the
	// recipe's internal RuntimeScope at the request/runtime seams instead.
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
// model name from the Entrypoint table. Such names never reach a backend.
func (c *RouterConfig) IsEntrypointModelName(modelName string) bool {
	_, ok := c.RecipeForRequestModel(modelName)
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
	if c.RoutingScope != "" {
		return c.Decisions
	}
	if len(c.Recipes) == 1 {
		return c.Recipes[0].Profile.Decisions
	}
	all := make([]Decision, 0)
	for i := range c.Recipes {
		all = append(all, c.Recipes[i].Profile.Decisions...)
	}
	return all
}

// HasRoutingDecisions reports whether the scoped view or any explicit Recipe
// declares decisions without allocating the aggregate inventory.
func (c *RouterConfig) HasRoutingDecisions() bool {
	if c == nil {
		return false
	}
	if c.RoutingScope != "" {
		return len(c.Decisions) > 0
	}
	for i := range c.Recipes {
		if len(c.Recipes[i].Profile.Decisions) > 0 {
			return true
		}
	}
	return false
}
