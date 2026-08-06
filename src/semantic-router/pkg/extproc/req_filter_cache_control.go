package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const defaultCacheControlHeader = "x-vsr-cache-control"

func applyRequestCacheControls(ctx *RequestContext) {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return
	}
	pluginConfig := ctx.VSRSelectedDecision.GetSemanticCacheConfig()
	if pluginConfig == nil || !pluginConfig.AllowRequestControls {
		return
	}
	headerName := strings.TrimSpace(pluginConfig.ControlHeader)
	if headerName == "" {
		headerName = defaultCacheControlHeader
	}
	value := strings.ToLower(strings.TrimSpace(headerValueCI(ctx, headerName)))
	switch value {
	case "no-cache":
		ctx.CacheReadBypass = true
	case "no-store":
		ctx.CacheWriteBypass = true
	case "bypass", "no-cache,no-store", "no-store,no-cache":
		ctx.CacheReadBypass = true
		ctx.CacheWriteBypass = true
	default:
		return
	}
	logging.ComponentDebugEvent("extproc", "response_cache_request_control", map[string]interface{}{
		"request_id":   ctx.RequestID,
		"read_bypass":  ctx.CacheReadBypass,
		"write_bypass": ctx.CacheWriteBypass,
	})
}
