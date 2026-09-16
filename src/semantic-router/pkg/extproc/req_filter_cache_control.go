package extproc

import (
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const defaultCacheControlHeader = "x-vsr-cache-control"

func applyRequestCacheControls(ctx *RequestContext) {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return
	}
	pluginConfig := ctx.VSRSelectedDecision.GetResponseCacheConfig()
	if pluginConfig == nil {
		return
	}
	controls := pluginConfig.EffectiveRequestControls()
	if !controls.Enabled {
		return
	}
	headerName := strings.TrimSpace(controls.Header)
	if headerName == "" {
		headerName = defaultCacheControlHeader
	}
	allowed := responseCacheAllowedControls(controls.Allowed)
	for _, rawDirective := range strings.Split(strings.ToLower(headerValueCI(ctx, headerName)), ",") {
		directive := strings.TrimSpace(rawDirective)
		name, value, hasValue := strings.Cut(directive, "=")
		if _, ok := allowed[name]; !ok {
			continue
		}
		applyResponseCacheDirective(ctx, controls.MaxTTLSeconds, name, value, hasValue)
	}
	logging.ComponentDebugEvent("extproc", "response_cache_request_control", map[string]interface{}{
		"request_id":   ctx.RequestID,
		"read_bypass":  ctx.CacheReadBypass,
		"write_bypass": ctx.CacheWriteBypass,
	})
}

func applyResponseCacheDirective(
	ctx *RequestContext,
	maxTTL *int,
	name string,
	value string,
	hasValue bool,
) {
	switch name {
	case "no-cache":
		ctx.CacheReadBypass = true
	case "no-store":
		ctx.CacheWriteBypass = true
	case "bypass":
		ctx.CacheReadBypass = true
		ctx.CacheWriteBypass = true
	case "max-age":
		if hasValue {
			ctx.CacheMaxAgeSeconds = parseNonNegativeCacheDirective(value)
		}
	case "ttl":
		if hasValue {
			ttl := parseNonNegativeCacheDirective(value)
			ctx.CacheWriteTTLSeconds = clampResponseCacheTTL(ttl, maxTTL)
		}
	}
}

func responseCacheAllowedControls(configured []string) map[string]struct{} {
	if len(configured) == 0 {
		configured = []string{"no-cache", "no-store", "bypass"}
	}
	allowed := make(map[string]struct{}, len(configured))
	for _, directive := range configured {
		allowed[strings.TrimSpace(strings.ToLower(directive))] = struct{}{}
	}
	return allowed
}

func parseNonNegativeCacheDirective(value string) *int {
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed < 0 {
		return nil
	}
	return &parsed
}

func clampResponseCacheTTL(ttl *int, maximum *int) *int {
	if ttl == nil || maximum == nil || *ttl <= *maximum {
		return ttl
	}
	clamped := *maximum
	return &clamped
}

func applyResponseCacheRequestTTL(ctx *RequestContext, configured int) int {
	if ctx == nil || ctx.CacheWriteTTLSeconds == nil {
		return configured
	}
	requested := *ctx.CacheWriteTTLSeconds
	if configured <= 0 || requested < configured {
		return requested
	}
	return configured
}
