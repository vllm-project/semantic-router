package extproc

import (
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// headerValueCI returns the first non-empty value for a header name using
// case-insensitive name matching.
func headerValueCI(ctx *RequestContext, canonical string) string {
	if ctx == nil || len(ctx.Headers) == 0 || canonical == "" {
		return ""
	}
	if v, ok := ctx.Headers[canonical]; ok && v != "" {
		return v
	}
	for k, v := range ctx.Headers {
		if strings.EqualFold(k, canonical) && v != "" {
			return v
		}
	}
	return ""
}

// authHeaderUserID returns the authenticated user id from the authz header.
// Matching is case-insensitive on the header name: Envoy/HTTP2 may normalize
// keys differently than our canonical constant, and direct map lookup would miss.
func authHeaderUserID(ctx *RequestContext) string {
	return headerValueCI(ctx, headers.AuthzUserID)
}

// cacheScopeUserID resolves the user id used only for semantic-cache key scoping.
// It prefers the trusted auth header, then optionally a fallback header name from
// SEMANTIC_CACHE_FALLBACK_USER_HEADER (intended for E2E when the gateway strips
// x-authz-user-id before extproc). Do not set that env in production.
func cacheScopeUserID(ctx *RequestContext) string {
	if u := authHeaderUserID(ctx); u != "" {
		return u
	}
	fallback := strings.TrimSpace(os.Getenv("SEMANTIC_CACHE_FALLBACK_USER_HEADER"))
	if fallback == "" {
		return ""
	}
	return headerValueCI(ctx, fallback)
}
