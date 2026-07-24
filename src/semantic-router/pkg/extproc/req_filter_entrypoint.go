package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// resolveEntrypointForRequest maps the inbound request model name through the
// configured entrypoint table before any signal evaluation, so decision
// evaluation and model routing observe the per-request recipe (issue #2331).
// Requests whose model matches no entrypoint keep a nil recipe and flow
// through the existing auto-model / specified-model handling.
func (r *OpenAIRouter) resolveEntrypointForRequest(originalModel string, ctx *RequestContext) {
	if r == nil || r.Config == nil || ctx == nil {
		return
	}
	recipe, ok := r.Config.RecipeForRequestModel(originalModel)
	if !ok {
		return
	}
	ctx.EntrypointRecipe = recipe
	logging.ComponentDebugEvent("extproc", "entrypoint_recipe_resolved", map[string]interface{}{
		"request_id": ctx.RequestID,
		"model":      originalModel,
		"recipe":     recipe.Name,
	})
}

// requestModelActsAsAuto reports whether the inbound model name is resolved by
// the router (auto slugs and entrypoint virtual names) rather than forwarded
// as a concrete backend model.
func (r *OpenAIRouter) requestModelActsAsAuto(modelName string) bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.IsAutoModelName(modelName) || r.Config.IsEntrypointModelName(modelName)
}

// decisionCandidatesForRequest scopes decision evaluation to the recipe the
// entrypoint table selected for this request. Requests outside the entrypoint
// table — and entrypoint aliases of the default recipe — keep the existing
// candidate behavior: algorithm virtual slugs filter by algorithm type and
// every other name evaluates the default recipe's decisions.
func (r *OpenAIRouter) decisionCandidatesForRequest(originalModel string, ctx *RequestContext) []config.Decision {
	if ctx != nil && ctx.EntrypointRecipe != nil && ctx.EntrypointRecipe.Name != config.DefaultRecipeName {
		return ctx.EntrypointRecipe.Decisions
	}
	return r.decisionCandidatesForRequestModel(originalModel)
}
