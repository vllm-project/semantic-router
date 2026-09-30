package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// replayConfigForRequest resolves the standing policy independently of a
// matched decision. Shared runtime services retain their existing settings.
func (r *OpenAIRouter) replayConfigForRequest(ctx *RequestContext) *config.RouterConfig {
	if r == nil || r.Config == nil {
		return nil
	}
	if ctx != nil {
		if recipe := ctx.Routing.SelectedRecipe(); recipe != nil {
			return r.Config.ConfigForRecipe(recipe)
		}
		if ctx.Routing.IsPassthrough() {
			// A concrete backend request does not select the default recipe.
			unscoped := *r.Config
			unscoped.DataPolicy = nil
			return &unscoped
		}
	}
	return r.Config
}

func (r *OpenAIRouter) replayAllowedForRequest(ctx *RequestContext) bool {
	if r == nil {
		return false
	}
	if ctx != nil {
		if recipe := ctx.Routing.SelectedRecipe(); recipe != nil {
			return recipe.Profile.DataPolicy.ReplayAllowed()
		}
		if ctx.Routing.IsPassthrough() {
			return true
		}
	}
	return r.Config == nil || r.Config.DataPolicy.ReplayAllowed()
}

func (r *OpenAIRouter) effectiveReplayConfigForRequest(ctx *RequestContext, decision *config.Decision) *config.RouterReplayPluginConfig {
	if !r.replayAllowedForRequest(ctx) {
		return nil
	}
	return r.replayConfigForRequest(ctx).EffectiveRouterReplayConfig(decision)
}

// Startup has recipe-qualified decision references but no request context.
// Resolve each profile before considering a shared or isolated store.
func replayConfigForDecisionRef(cfg *config.RouterConfig, ref config.RoutingDecisionRef) *config.RouterReplayPluginConfig {
	if recipe, ok := cfg.RecipeByName(ref.Recipe); ok {
		cfg = cfg.ConfigForRecipe(recipe)
	}
	return cfg.EffectiveRouterReplayConfig(ref.Decision)
}
