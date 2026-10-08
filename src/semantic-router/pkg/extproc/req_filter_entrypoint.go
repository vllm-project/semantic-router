package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// resolveEntrypointForRequest resolves the routing profile before any signal
// evaluation. Every virtual entrypoint selects its mapped recipe; concrete
// backend models keep a
// nil recipe so they bypass recipe routing entirely.
func (r *OpenAIRouter) resolveEntrypointForRequest(originalModel string, ctx *RequestContext) {
	if r == nil || r.Config == nil || ctx == nil {
		return
	}
	recipe, ok := r.Config.RecipeForRoutingModel(originalModel)
	if !ok {
		ctx.Routing.SelectPassthrough()
		return
	}
	ctx.Routing.SelectRecipe(recipe)
	observeRoutingIdentity(ctx, originalModel)
	logging.ComponentDebugEvent("extproc", "entrypoint_recipe_resolved", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      originalModel,
		"recipe":     recipe.Name,
	})
}

func (r *OpenAIRouter) classifierForRequest(ctx *RequestContext) *classification.Classifier {
	if r == nil || ctx == nil || ctx.Routing.SelectedRecipe() == nil {
		return nil
	}
	recipe := ctx.Routing.SelectedRecipe()
	// Programmatic single-profile routers may provide only the default
	// classifier. Named recipes never fall back across the isolation boundary.
	if r.RecipeClassifiers == nil {
		if recipe.Name == config.DefaultRecipeName {
			return r.Classifier
		}
		return nil
	}
	classifier, ok := r.RecipeClassifiers.ForRecipe(recipe.Name)
	if !ok {
		return nil
	}
	return classifier
}

// requestModelActsAsAuto reports whether the inbound model name is resolved by
// the router through the effective entrypoint table rather than forwarded
// as a concrete backend model.
func (r *OpenAIRouter) requestModelActsAsAuto(modelName string) bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.IsEntrypointModelName(modelName)
}

// decisionCandidatesForRequest scopes evaluation to exactly one recipe.
func (r *OpenAIRouter) decisionCandidatesForRequest(originalModel string, ctx *RequestContext) []config.Decision {
	if ctx != nil && ctx.Routing.SelectedRecipe() != nil {
		recipe := ctx.Routing.SelectedRecipe()
		if recipe.Profile.Decisions == nil {
			// A recipe with no decisions still scopes evaluation: an empty,
			// non-nil slice keeps runDecisionEngine from falling back to the
			// default profile's decisions.
			return []config.Decision{}
		}
		return recipe.Profile.Decisions
	}
	return []config.Decision{}
}

// decisionCandidatesForRequestModel resolves every routing alias through the
// same recipe table; algorithms are selected by decisions, never model strings.
func (r *OpenAIRouter) decisionCandidatesForRequestModel(modelName string) []config.Decision {
	if r == nil || r.Config == nil {
		return nil
	}
	recipe, ok := r.Config.RecipeForRoutingModel(modelName)
	if !ok {
		return nil
	}
	return recipe.Profile.Decisions
}
