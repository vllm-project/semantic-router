package config

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const standaloneRoutingRevision int64 = 1

// CompileStandaloneRoutingSnapshot is the only human-authoring to strict
// runtime boundary. It resolves readable names and Provider Integrations once,
// then returns the same immutable snapshot consumed in managed mode.
func CompileStandaloneRoutingSnapshot(
	canonical CanonicalConfig,
	connectionCompiler modelauthoring.ConnectionCompiler,
) (*routingsnapshot.Snapshot, error) {
	models, modelsByName, err := compileAuthoringModels(context.Background(), canonical.Models, connectionCompiler)
	if err != nil {
		return nil, err
	}
	recipes, recipesByName, err := compileAuthoringRecipes(canonical.Recipes)
	if err != nil {
		return nil, err
	}
	entrypoints, err := compileAuthoringEntrypoints(canonical.Entrypoints, recipesByName, modelsByName)
	if err != nil {
		return nil, err
	}

	identityPayload, err := json.Marshal(struct {
		Models      []AuthoringModel
		Recipes     []AuthoringRecipe
		Entrypoints []AuthoringEntrypoint
	}{canonical.Models, canonical.Recipes, canonical.Entrypoints})
	if err != nil {
		return nil, fmt.Errorf("encode standalone routing identity: %w", err)
	}
	currency, _ := canonicalBillingCurrency(&canonical)
	bundle := routingsnapshot.Bundle{
		NamespaceID: uuid.NewSHA1(
			uuid.NameSpaceOID,
			append([]byte("vllm-sr/standalone-routing/v1\x00"), identityPayload...),
		).String(),
		Revision: standaloneRoutingRevision, Currency: currency,
		Models: models, Recipes: recipes, Entrypoints: entrypoints,
	}
	verified, err := routingsnapshot.Compile(bundle)
	if err != nil {
		return nil, fmt.Errorf("compile standalone routing snapshot: %w", err)
	}
	return verified, nil
}

func compileAuthoringRecipes(
	sources []AuthoringRecipe,
) ([]routingsnapshot.Recipe, map[string]routingsnapshot.Recipe, error) {
	result := make([]routingsnapshot.Recipe, 0, len(sources))
	byName := make(map[string]routingsnapshot.Recipe, len(sources))
	for _, source := range sources {
		recipeID := stableRoutingResourceID("rcp", source.Name)
		if RecipeName(source.Name) == DefaultRecipeName {
			recipeID = "rcp_default"
		}
		document := source.Document
		document.Decisions = cloneEntrypointDecisions(source.Document.Decisions)
		decisions := make([]routingsnapshot.Decision, 0, len(document.Decisions))
		for index := range document.Decisions {
			decision := &document.Decisions[index]
			algorithmType := ""
			if decision.Algorithm != nil {
				algorithmType = decision.Algorithm.Type
			}
			cardinality, known := DecisionAlgorithmDispatchCardinality(algorithmType)
			if !known {
				return nil, nil, fmt.Errorf("recipe %q decision %q has unknown algorithm %q", source.Name, decision.Name, algorithmType)
			}
			decisions = append(decisions, routingsnapshot.Decision{
				ID: stableRoutingResourceID("dec", recipeID, decision.Name), Name: decision.Name,
				DispatchCardinality: routingsnapshot.DispatchCardinality(cardinality),
			})
		}
		documentJSON, err := MarshalManagedRecipeDocument(document)
		if err != nil {
			return nil, nil, fmt.Errorf("recipe %q: %w", source.Name, err)
		}
		compiled := routingsnapshot.Recipe{
			ID: recipeID, Revision: initialRoutingResourceRevision,
			Name: source.Name, Description: source.Description,
			Decisions: decisions, Document: documentJSON,
		}
		result = append(result, compiled)
		byName[source.Name] = compiled
	}
	return result, byName, nil
}

func compileAuthoringEntrypoints(
	sources []AuthoringEntrypoint,
	recipesByName map[string]routingsnapshot.Recipe,
	modelsByName map[string]routingsnapshot.Model,
) ([]routingsnapshot.Entrypoint, error) {
	result := make([]routingsnapshot.Entrypoint, 0, len(sources))
	for entrypointIndex, source := range sources {
		entrypointID := stableRoutingResourceID("ep", source.Name)
		rules, err := authoringEntrypointRules(source)
		if err != nil {
			return nil, fmt.Errorf("entrypoints[%d]: %w", entrypointIndex, err)
		}
		compiled := routingsnapshot.Entrypoint{
			ID: entrypointID, Revision: initialRoutingResourceRevision, Name: source.Name,
			Aliases: stableUniqueStrings(append([]string{source.Name}, source.Aliases...)),
			Rules:   make([]routingsnapshot.EntrypointRule, 0, len(rules)),
		}
		for ruleIndex, sourceRule := range rules {
			recipe, found := recipesByName[strings.TrimSpace(sourceRule.Recipe)]
			if !found {
				return nil, fmt.Errorf("entrypoints[%d].rules[%d].recipe references unknown Recipe %q", entrypointIndex, ruleIndex, sourceRule.Recipe)
			}
			matchers, err := compileAuthoringMatchers(sourceRule.Matches)
			if err != nil {
				return nil, fmt.Errorf("entrypoints[%d].rules[%d].matches: %w", entrypointIndex, ruleIndex, err)
			}
			assignments, err := compileAuthoringAssignments(sourceRule.Assignments, recipe, modelsByName)
			if err != nil {
				return nil, fmt.Errorf("entrypoints[%d].rules[%d].assignments: %w", entrypointIndex, ruleIndex, err)
			}
			compiled.Rules = append(compiled.Rules, routingsnapshot.EntrypointRule{
				ID: stableRoutingResourceID("rule", entrypointID, sourceRule.Name), Name: sourceRule.Name,
				Matchers: matchers, RecipeID: recipe.ID, RecipeRevision: recipe.Revision,
				Assignments: assignments,
			})
		}
		result = append(result, compiled)
	}
	return result, nil
}

func compileAuthoringMatchers(input []AuthoringEntrypointMatch) ([]routingsnapshot.Matcher, error) {
	normalized, err := normalizeEntrypointMatches("matches", input)
	if err != nil {
		return nil, err
	}
	result := make([]routingsnapshot.Matcher, 0, len(normalized))
	for _, matcher := range normalized {
		switch {
		case matcher.Claim != nil:
			result = append(result, routingsnapshot.Matcher{Claim: &routingsnapshot.ClaimMatcher{
				Name: matcher.Claim.Name,
				Value: routingsnapshot.ClaimValue{
					Kind: matcher.Claim.Value.Kind, String: matcher.Claim.Value.String,
					Boolean: matcher.Claim.Value.Boolean, Integer: matcher.Claim.Value.Integer,
				},
			}})
		case matcher.Path != nil && matcher.Path.Exact != "":
			result = append(result, routingsnapshot.Matcher{ExactPath: matcher.Path.Exact})
		case matcher.Path != nil:
			result = append(result, routingsnapshot.Matcher{PathPrefix: matcher.Path.Prefix})
		}
	}
	return result, nil
}

func compileAuthoringAssignments(
	input map[string]AuthoringAssignmentSet,
	recipe routingsnapshot.Recipe,
	modelsByName map[string]routingsnapshot.Model,
) (map[string]routingsnapshot.AssignmentSet, error) {
	decisionsByName := make(map[string]routingsnapshot.Decision, len(recipe.Decisions))
	for _, decision := range recipe.Decisions {
		decisionsByName[decision.Name] = decision
	}
	result := make(map[string]routingsnapshot.AssignmentSet, len(input))
	for decisionName, sourceSet := range input {
		decision, found := decisionsByName[decisionName]
		if !found {
			return nil, fmt.Errorf("%q references unknown Decision name", decisionName)
		}
		set := routingsnapshot.AssignmentSet{Models: make([]routingsnapshot.Assignment, 0, len(sourceSet.Models))}
		for modelIndex, source := range sourceSet.Models {
			model, found := modelsByName[strings.TrimSpace(source.Model)]
			if !found {
				return nil, fmt.Errorf("%s.models[%d].model references unknown Model %q", decisionName, modelIndex, source.Model)
			}
			var reasoning *routingsnapshot.AssignmentReasoning
			if source.Reasoning != nil {
				reasoning = &routingsnapshot.AssignmentReasoning{
					Enabled: source.Reasoning.Enabled, Effort: source.Reasoning.Effort,
					Description: source.Reasoning.Description,
				}
			}
			set.Models = append(set.Models, routingsnapshot.Assignment{
				ModelID: model.ID, ModelRevision: model.Revision,
				Priority: source.Priority, Weight: source.Weight,
				LoRAName: source.LoRAName, Reasoning: reasoning,
			})
		}
		if sourceSet.Fallback != nil {
			set.Fallback = &routingsnapshot.FallbackPolicy{
				Strategy: sourceSet.Fallback.Strategy,
				On:       append([]string(nil), sourceSet.Fallback.On...),
			}
		}
		result[decision.ID] = set
	}
	return result, nil
}
