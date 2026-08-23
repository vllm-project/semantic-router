package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// resolveEntrypointForRequest resolves the routing profile before any signal
// evaluation. Only an explicit Entrypoint selects a Recipe; concrete backend
// Models and unknown names keep a nil Recipe and bypass Recipe routing.
func (r *OpenAIRouter) resolveEntrypointForRequest(originalModel string, ctx *RequestContext) {
	if r == nil || r.Config == nil || ctx == nil {
		return
	}
	recipe, ok := r.Config.RecipeForRequestModel(originalModel)
	if !ok {
		ctx.Routing.SelectPassthrough()
		return
	}
	ctx.Routing.SelectRecipe(recipe)
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
	// Entrypoint bindings only replace physical decision targets. Signals and
	// projections remain the reusable recipe's read-only policy, so derived
	// views intentionally share its compiled classifier. Mutable selection,
	// replay, learning, cache, and session state are keyed by RuntimeScope.
	if r.RecipeClassifiers == nil {
		return nil
	}
	classifier, ok := r.RecipeClassifiers.ForRecipe(recipe.Name)
	if !ok {
		return nil
	}
	return classifier
}

// requestModelIsEntrypoint reports whether the inbound Model is an explicit
// request-facing Entrypoint rather than a concrete backend Model.
func (r *OpenAIRouter) requestModelIsEntrypoint(modelName string) bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.IsEntrypointModelName(modelName)
}

// decisionCandidatesForRequest scopes evaluation to exactly one Recipe.
func (r *OpenAIRouter) decisionCandidatesForRequest(ctx *RequestContext) []config.Decision {
	if ctx != nil && ctx.Routing.SelectedRecipe() != nil {
		recipe := ctx.Routing.SelectedRecipe()
		if recipe.Profile.Decisions == nil {
			// A recipe with no decisions still scopes evaluation: an empty,
			// non-nil slice keeps runDecisionEngine from escaping the selected
			// Recipe.
			return []config.Decision{}
		}
		return recipe.Profile.Decisions
	}
	return []config.Decision{}
}
