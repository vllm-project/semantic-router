package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// requestPlugins are the plugins a request runs: its decision's; for a
// request that matched no decision, its recipe's default plugins
// (global.plugins layered with the recipe's); for a passthrough request,
// global.plugins.
func requestPlugins(cfg *config.RouterConfig, ctx *RequestContext) []config.DecisionPlugin {
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return ctx.VSRSelectedDecision.Plugins
	}
	if ctx != nil {
		if recipe := ctx.Routing.SelectedRecipe(); recipe != nil {
			return recipe.Profile.DefaultPlugins
		}
	}
	if cfg == nil {
		return nil
	}
	return cfg.GlobalPlugins
}

// requestResponseCache is the response_cache plugin a routed request runs; a
// passthrough request bypasses the response cache.
func requestResponseCache(ctx *RequestContext) *config.ResponseCachePluginConfig {
	return config.ResponseCachePluginOf(requestPlugins(nil, ctx))
}

// requestMasking is the masking plugin a routed request runs. It masks what a
// pii rule of the request's recipe finds, so a passthrough request, which has
// no recipe, runs none.
func requestMasking(ctx *RequestContext) *config.MaskingPluginConfig {
	return config.MaskingPluginOf(requestPlugins(nil, ctx))
}

// requestResponseGuard is the response_guard plugin a routed request runs. Its
// rules read the request's recipe signals, which a passthrough request
// bypasses, so it runs none.
func requestResponseGuard(ctx *RequestContext) *config.ResponseGuardPluginConfig {
	return config.ResponseGuardPluginOf(requestPlugins(nil, ctx))
}
