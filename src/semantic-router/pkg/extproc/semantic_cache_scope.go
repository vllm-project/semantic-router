package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (r *OpenAIRouter) semanticCacheEnabledForRequest(ctx *RequestContext) bool {
	if r == nil || r.Config == nil {
		return true
	}
	if requestBypassesRouting(ctx) {
		return false
	}
	if responseCacheScope(ctx) != "global" &&
		strings.TrimSpace(responseCacheScopeIdentity(ctx)) == "" {
		return false
	}
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		return r.Config.IsCacheEnabledForDecisionObject(ctx.VSRSelectedDecision)
	}
	if r.Config.HasRoutingDecisions() {
		return false
	}
	return r.Config.Enabled
}

func responseCacheModeForRequest(ctx *RequestContext) string {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return config.ResponseCacheModeSemantic
	}
	pluginConfig := ctx.VSRSelectedDecision.GetResponseCacheConfig()
	if pluginConfig == nil || strings.TrimSpace(pluginConfig.Mode) == "" {
		return config.ResponseCacheModeSemantic
	}
	return strings.TrimSpace(pluginConfig.Mode)
}

func exactCacheEnabledForRequest(ctx *RequestContext) bool {
	mode := responseCacheModeForRequest(ctx)
	return mode == config.ResponseCacheModeExact ||
		mode == config.ResponseCacheModeExactThenSemantic
}

func semanticLookupEnabledForRequest(ctx *RequestContext) bool {
	return responseCacheModeForRequest(ctx) != config.ResponseCacheModeExact
}
