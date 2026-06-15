//go:build !dev

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// extractUserID extracts user ID from the trusted auth header only.
//
// PRODUCTION BUILD: This version ONLY uses the auth header (x-authz-user-id)
// injected by the external auth service. There is NO fallback to metadata["user_id"]
// to prevent bypassing the authorization gate.
//
// The auth header is injected by the external auth service (Authorino, Envoy Gateway JWT,
// oauth2-proxy, etc.) and is the only trusted source for user identity.
func extractUserID(ctx *RequestContext) string {
	// Check auth header (trusted source, injected by auth backend)
	if userID, ok := ctx.Headers[headers.AuthzUserID]; ok && userID != "" {
		logging.ComponentDebugEvent("extproc", "memory_user_id_resolved", map[string]interface{}{
			"request_id": ctx.RequestID,
			"source":     "auth_header",
			"header":     headers.AuthzUserID,
		})
		return userID
	}

	// PRODUCTION: No fallback - auth header is the only trusted source
	return ""
}
