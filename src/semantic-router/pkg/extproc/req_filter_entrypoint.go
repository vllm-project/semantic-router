package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// deniedEntrypointReason is the external error message for a claimed
// conditional entrypoint alias with no matching rule. It is deliberately
// identical to an unknown-model response: distinguishing "hidden" from
// "doesn't exist" would let a caller enumerate other tenants' entrypoints.
const deniedEntrypointReason = "The model does not exist or is not available for this request context."

// resolveEntrypointForRequest resolves the routing profile before any signal
// evaluation. Auto and direct-looper aliases select the default recipe;
// legacy entrypoints select their mapped recipe unconditionally; conditional
// entrypoints (Rules) are evaluated against the caller's headers and the
// request path, and a claimed-but-unmatched alias is denied — it must never
// fall through to passthrough or the default recipe. Concrete backend models
// (unclaimed by any entrypoint) keep a nil recipe so they bypass recipe
// routing entirely.
func (r *OpenAIRouter) resolveEntrypointForRequest(originalModel string, ctx *RequestContext) {
	if r == nil || r.Config == nil || ctx == nil {
		return
	}
	trimmed := strings.TrimSpace(originalModel)

	if r.Config.IsAutoModelName(trimmed) || r.Config.IsReMoMModelName(trimmed) ||
		r.Config.IsFusionModelName(trimmed) || r.Config.IsFlowModelName(trimmed) {
		recipe := r.Config.DefaultRecipe()
		if recipe == nil {
			ctx.Routing.SelectPassthrough()
			return
		}
		ctx.Routing.SelectRecipe(recipe)
		return
	}

	entrypoint, ok := r.Config.EntrypointByModelName(trimmed)
	if !ok {
		ctx.Routing.SelectPassthrough()
		return
	}

	if len(entrypoint.Rules) == 0 {
		recipe, ok := r.Config.RecipeByName(entrypoint.Recipe)
		if !ok {
			// Unreachable once config validation requires every legacy
			// entrypoint to reference a known recipe; fail closed rather
			// than assume, in case this is ever reached anyway.
			logging.Errorf("[Entrypoint] entrypoint %q references unknown recipe %q", trimmed, entrypoint.Recipe)
			ctx.Routing.SelectDenied(500, "internal routing configuration error")
			return
		}
		ctx.Routing.SelectRecipe(recipe)
		logging.ComponentDebugEvent("extproc", "entrypoint_recipe_resolved", map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      trimmed,
			"recipe":     recipe.Name,
		})
		return
	}

	matchCtx := config.MatchContext{
		Path:    config.NormalizeRequestPath(ctx.Headers[":path"]),
		Headers: ctx.Headers,
	}
	resolution := r.Config.ResolveEntrypoint(trimmed, matchCtx)
	switch resolution.Status {
	case config.EntrypointMatched:
		ctx.Routing.SelectRecipe(resolution.Recipe)
		fields := map[string]interface{}{
			"request_id": ctx.RequestID,
			"model":      trimmed,
			"recipe":     resolution.Recipe.Name,
		}
		if resolution.Rule != nil {
			fields["rule"] = resolution.Rule.Name
		}
		logging.ComponentDebugEvent("extproc", "entrypoint_recipe_resolved", fields)
	case config.EntrypointClaimedNoMatch:
		ctx.Routing.SelectDenied(404, deniedEntrypointReason)
	default:
		// EntrypointAmbiguous or EntrypointUnclaimed: both unreachable here
		// (Unclaimed already returned above; Ambiguous config is rejected at
		// validation time). Fail closed, not open, if reached anyway.
		logging.Errorf("[Entrypoint] unexpected resolution status %v for claimed model %q", resolution.Status, trimmed)
		ctx.Routing.SelectDenied(500, "internal routing configuration error")
	}
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
