//go:build dev

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// extractUserID extracts user ID with priority: auth header > metadata fallback.
//
// DEV BUILD: This development version includes fallbacks for development/testing.
// These are UNTRUSTED (client-provided) and should ONLY be used for development/testing.
//
// Priority 1: Auth header (x-authz-user-id) injected by the external auth service
// (Authorino, Envoy Gateway JWT, oauth2-proxy, etc.). This is the trusted source.
//
// Priority 2: metadata["user_id"] from Response API request body (untrusted).
//
// Priority 3: Chat Completions user ID - extracted with its own priority:
//   - metadata["user_id"] (consistent with Response API)
//   - "user" field (deprecated by OpenAI, kept for backward compatibility)
func extractUserID(ctx *RequestContext) string {
	// Check auth header first (trusted source, injected by auth backend)
	if userID, ok := ctx.Headers[headers.AuthzUserID]; ok && userID != "" {
		logging.ComponentDebugEvent("extproc", "memory_user_id_resolved", map[string]interface{}{
			"request_id": ctx.RequestID,
			"source":     "auth_header",
			"header":     headers.AuthzUserID,
		})
		return userID
	}

	// DEV-ONLY: Fallback to metadata["user_id"] (untrusted, for development/testing)
	if ctx.SemanticRequest != nil {
		if ctx.SemanticRequest.Metadata != nil {
			if userID, ok := ctx.SemanticRequest.Metadata["user_id"]; ok && userID != "" {
				logging.ComponentWarnEvent("extproc", "memory_user_id_untrusted_fallback", map[string]interface{}{
					"request_id": ctx.RequestID,
					"source":     "response_api_metadata",
				})
				return userID
			}
		}
	}

	return ""
}
