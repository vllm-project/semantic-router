package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func trustedIdentityUserID(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	userID := strings.TrimSpace(ctx.TrustedIdentity.UserID)
	if userID == "" {
		return ""
	}
	logging.ComponentDebugEvent("extproc", "memory_user_id_resolved", map[string]interface{}{
		"request_id": ctx.RequestID,
		"source":     "trusted_identity",
	})
	return userID
}

// cacheScopeUserID resolves the user id used for semantic-cache key scoping.
// Identity is ingress-derived once and is never reconstructed from request
// headers, body metadata, or environment-controlled fallback headers.
func cacheScopeUserID(ctx *RequestContext) string {
	requestID := ""
	if ctx != nil {
		requestID = ctx.RequestID
		if u := strings.TrimSpace(ctx.TrustedIdentity.UserID); u != "" {
			logging.ComponentDebugEvent("extproc", "cache_scope_user_resolved", map[string]interface{}{
				"request_id": ctx.RequestID,
				"source":     "trusted_identity",
			})
			return u
		}
	}
	logging.ComponentDebugEvent("extproc", "cache_scope_user_missing", map[string]interface{}{
		"request_id": requestID,
	})
	return ""
}

func responseCacheScope(ctx *RequestContext) string {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return "global"
	}
	plugin := ctx.VSRSelectedDecision.GetResponseCacheConfig()
	if plugin == nil {
		return "global"
	}
	scope := strings.TrimSpace(plugin.Scope)
	if scope == "" {
		return "user"
	}
	return scope
}

func responseCacheScopeIdentity(ctx *RequestContext) string {
	switch responseCacheScope(ctx) {
	case "global":
		return ""
	case "tenant":
		if ctx == nil {
			return ""
		}
		return strings.TrimSpace(ctx.TrustedIdentity.TenantID)
	case "team":
		if ctx == nil {
			return ""
		}
		if team := strings.TrimSpace(ctx.TrustedIdentity.TeamID); team != "" {
			return team
		}
		if len(ctx.TrustedIdentity.Groups) > 0 {
			return strings.TrimSpace(ctx.TrustedIdentity.Groups[0])
		}
		return ""
	default:
		return cacheScopeUserID(ctx)
	}
}
