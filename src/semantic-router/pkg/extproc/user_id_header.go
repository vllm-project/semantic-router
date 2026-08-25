package extproc

import "strings"

// headerValueCI is limited to protocol and routing headers. Identity helpers
// below never call it: authenticated identity comes only from authenticated state.
func headerValueCI(ctx *RequestContext, canonical string) string {
	if ctx == nil || len(ctx.Headers) == 0 || canonical == "" {
		return ""
	}
	if value, ok := ctx.Headers[canonical]; ok && value != "" {
		return value
	}
	for key, value := range ctx.Headers {
		if strings.EqualFold(key, canonical) && value != "" {
			return value
		}
	}
	return ""
}

func cacheScopeUserID(ctx *RequestContext) string {
	if ctx == nil || ctx.InferenceAccess == nil {
		return ""
	}
	return strings.TrimSpace(ctx.InferenceAccess.tenant.UserID)
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
	if ctx == nil || ctx.InferenceAccess == nil {
		return ""
	}
	tenant := ctx.InferenceAccess.tenant
	switch responseCacheScope(ctx) {
	case "global":
		return ""
	case "tenant":
		return strings.TrimSpace(tenant.NamespaceID)
	case "team":
		return strings.TrimSpace(tenant.TeamID)
	default:
		return strings.TrimSpace(tenant.UserID)
	}
}
