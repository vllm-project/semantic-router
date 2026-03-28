package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// authHeaderUserID returns the authenticated user id from the authz header.
// Matching is case-insensitive on the header name: Envoy/HTTP2 may normalize
// keys differently than our canonical constant, and direct map lookup would miss.
func authHeaderUserID(ctx *RequestContext) string {
	if ctx == nil || len(ctx.Headers) == 0 {
		return ""
	}
	if v, ok := ctx.Headers[headers.AuthzUserID]; ok && v != "" {
		return v
	}
	for k, v := range ctx.Headers {
		if strings.EqualFold(k, headers.AuthzUserID) && v != "" {
			return v
		}
	}
	return ""
}
