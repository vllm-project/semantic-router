//go:build !dev

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// extractUserID extracts user ID from the trusted auth header only.
//
// PRODUCTION BUILD: This version ONLY uses the auth header (the configured
// global.services.authz.identity.user_id_header, x-authz-user-id by default)
// injected by the external auth service. There is NO fallback to metadata["user_id"]
// to prevent bypassing the authorization gate.
//
// The auth header is injected by the external auth service (Authorino, Envoy Gateway JWT,
// oauth2-proxy, etc.) and is the only trusted source for user identity.
func extractUserID(ctx *RequestContext) string {
	// Check auth header (trusted source, injected by auth backend)
	if userID := authHeaderUserID(ctx); userID != "" {
		logging.ComponentDebugEvent("extproc", "memory_user_id_resolved", map[string]interface{}{
			"request_id": ctx.RequestID,
			"source":     "auth_header",
			"header":     ctx.identity.GetUserIDHeader(),
		})
		return userID
	}

	// PRODUCTION: No fallback - auth header is the only trusted source
	return ""
}
