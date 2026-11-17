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
	plugin := requestResponseCache(ctx)
	return plugin != nil && plugin.Enabled
}

func responseCacheModeForRequest(ctx *RequestContext) string {
	pluginConfig := requestResponseCache(ctx)
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
