package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// resolveEntrypointForRequest resolves the routing profile before any signal
// evaluation. Auto and direct-looper aliases select the default recipe, named
// entrypoints select their mapped recipe, and concrete backend models keep a
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
// the router (auto slugs and entrypoint virtual names) rather than forwarded
// as a concrete backend model.
func (r *OpenAIRouter) requestModelActsAsAuto(modelName string) bool {
	if r == nil || r.Config == nil {
		return false
	}
	return r.Config.IsAutoModelName(modelName) || r.Config.IsEntrypointModelName(modelName)
}

// decisionCandidatesForRequest scopes evaluation to exactly one recipe. Direct
// looper aliases additionally filter the default recipe by algorithm type.
func (r *OpenAIRouter) decisionCandidatesForRequest(originalModel string, ctx *RequestContext) []config.Decision {
	if ctx != nil && ctx.Routing.SelectedRecipe() != nil {
		recipe := ctx.Routing.SelectedRecipe()
		if r.Config.IsReMoMModelName(originalModel) || r.Config.IsFusionModelName(originalModel) || r.Config.IsFlowModelName(originalModel) {
			return r.decisionCandidatesForRequestModel(originalModel)
		}
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

func (r *OpenAIRouter) decisionCandidatesForRequestModel(modelName string) []config.Decision {
	if r == nil || r.Config == nil {
		return nil
	}
	algorithm := ""
	switch {
	case r.Config.IsReMoMModelName(modelName):
		algorithm = config.DecisionAlgorithmReMoM
	case r.Config.IsFusionModelName(modelName):
		algorithm = config.DecisionAlgorithmFusion
	case r.Config.IsFlowModelName(modelName):
		algorithm = config.DecisionAlgorithmWorkflows
	default:
		return nil
	}
	candidates := make([]config.Decision, 0)
	for _, decision := range r.Config.Decisions {
		if decision.Algorithm != nil && decision.Algorithm.Type == algorithm {
			candidates = append(candidates, decision)
		}
	}
	return candidates
}
