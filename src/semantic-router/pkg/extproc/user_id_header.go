package extproc

import (
	"encoding/json"
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

func chatCompletionUserFieldFromBody(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	var req struct {
		User string `json:"user"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	return strings.TrimSpace(req.User)
}

// cacheScopeUserID resolves the user id used only for semantic-cache key scoping.
// It prefers the trusted auth header, then optionally a fallback header name from
// SEMANTIC_CACHE_FALLBACK_USER_HEADER (intended for E2E when the gateway strips
// x-authz-user-id before extproc). When SEMANTIC_CACHE_E2E_USER_FROM_BODY is "true",
// the OpenAI Chat Completions "user" field is used as a last resort (kubernetes E2E only).
// Do not set these env vars in production.
func cacheScopeUserID(ctx *RequestContext) string {
	if u := authHeaderUserID(ctx); u != "" {
		return u
	}
	fallback := strings.TrimSpace(os.Getenv("SEMANTIC_CACHE_FALLBACK_USER_HEADER"))
	if fallback != "" {
		if u := headerValueCI(ctx, fallback); u != "" {
			return u
		}
	}
	if strings.TrimSpace(os.Getenv("SEMANTIC_CACHE_E2E_USER_FROM_BODY")) == "true" {
		return chatCompletionUserFieldFromBody(ctx.OriginalRequestBody)
	}
	return ""
}
