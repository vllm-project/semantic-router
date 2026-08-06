package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func applyProviderPromptCachePlugin(
	ctx *RequestContext,
	passthrough *anthropic.AnthropicPassthrough,
) *anthropic.AnthropicPassthrough {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return passthrough
	}
	pluginConfig := ctx.VSRSelectedDecision.GetProviderPromptCacheConfig()
	if pluginConfig == nil || !pluginConfig.Enabled {
		return passthrough
	}
	if providerPromptCacheBypassed(
		ctx,
		pluginConfig.ControlHeader,
		pluginConfig.AllowRequestControls,
	) {
		return passthrough
	}
	if passthrough == nil {
		passthrough = &anthropic.AnthropicPassthrough{}
	}
	passthrough.AutoCacheControl = &anthropic.AutoCacheControlSpec{
		System:   pluginConfig.System,
		Tools:    pluginConfig.Tools,
		LastUser: pluginConfig.LastUser,
		TTL:      pluginConfig.TTL,
	}
	logging.ComponentDebugEvent("extproc", "provider_prompt_cache_applied", map[string]interface{}{
		"request_id": ctx.RequestID,
		"decision":   ctx.VSRSelectedDecision.Name,
		"system":     pluginConfig.System,
		"tools":      pluginConfig.Tools,
		"last_user":  pluginConfig.LastUser,
		"ttl":        pluginConfig.TTL,
	})
	return passthrough
}

func providerPromptCacheBypassed(
	ctx *RequestContext,
	controlHeader string,
	allowRequestControls bool,
) bool {
	if !allowRequestControls {
		return false
	}
	controlHeader = strings.TrimSpace(controlHeader)
	if controlHeader == "" {
		controlHeader = "x-vsr-provider-cache-control"
	}
	switch strings.ToLower(strings.TrimSpace(headerValueCI(ctx, controlHeader))) {
	case "bypass", "no-cache":
		return true
	default:
		return false
	}
}
