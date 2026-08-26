package managementserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func routingCatalogDTO(value accessmanagement.RoutingCatalog) managementapi.RoutingCatalog {
	result := managementapi.RoutingCatalog{
		KeyID: value.Subject.ID, PolicyRevision: value.PolicyRevision,
		PolicyDigest: value.PolicyDigest, RoutingRevision: value.RoutingRevision,
		RoutingDigest: value.RoutingDigest,
		Models:        make([]managementapi.RoutingCatalogModel, 0, len(value.Models)),
		Recipes:       make([]managementapi.RoutingCatalogRecipe, 0, len(value.Recipes)),
		Entrypoints:   make([]managementapi.RoutingCatalogEntrypoint, 0, len(value.Entrypoints)),
	}
	for _, model := range value.Models {
		result.Models = append(result.Models, managementapi.RoutingCatalogModel{
			ID: model.ID, Revision: model.Revision, Name: model.Name,
			Aliases: append([]string{}, model.Aliases...), ParamSize: model.ParamSize,
			ContextWindowSize: model.ContextWindowSize, Description: model.Description,
			Capabilities: append([]string{}, model.Capabilities...),
			Reasoning:    routingReasoningDTO(model.Reasoning), LoRAs: append([]string{}, model.LoRAs...),
			QualityScore: model.QualityScore, Modality: model.Modality, Tags: append([]string{}, model.Tags...),
			Pricing: managementapi.RoutingPricing(model.Pricing),
		})
	}
	for _, recipe := range value.Recipes {
		result.Recipes = append(result.Recipes, managementapi.RoutingCatalogRecipe{
			ID: recipe.ID, Revision: recipe.Revision, Name: recipe.Name,
			Description: recipe.Description, Decisions: routingDecisionsDTO(recipe.Decisions),
		})
	}
	for _, entrypoint := range value.Entrypoints {
		rules := make([]routingsnapshot.EntrypointRule, 0, len(entrypoint.Rules))
		for _, rule := range entrypoint.Rules {
			assignments := make(map[string]routingsnapshot.AssignmentSet, len(rule.Assignments))
			for decisionID, set := range rule.Assignments {
				assignments[decisionID] = routingsnapshot.AssignmentSet{
					Models:   append([]routingsnapshot.Assignment(nil), set.Models...),
					Fallback: set.Fallback,
				}
			}
			rules = append(rules, routingsnapshot.EntrypointRule{
				ID: rule.ID, Name: rule.Name, Matchers: rule.Matchers,
				RecipeID: rule.RecipeID, RecipeRevision: rule.RecipeRevision,
				Assignments: assignments,
			})
		}
		result.Entrypoints = append(result.Entrypoints, managementapi.RoutingCatalogEntrypoint{
			ID: entrypoint.ID, Revision: entrypoint.Revision, Name: entrypoint.Name,
			Aliases: append([]string{}, entrypoint.Aliases...), Rules: routingEntrypointRulesDTO(rules),
		})
	}
	return result
}
