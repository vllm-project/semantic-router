//go:build dev

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// extractUserID extracts user ID with priority: auth header > metadata fallback.
//
// DEV BUILD: This development version includes a fallback to metadata["user_id"].
// This is UNTRUSTED (client-provided) and should ONLY be used for development/testing.
//
// Priority 1: Auth header (x-authz-user-id) injected by the external auth service
// (Authorino, Envoy Gateway JWT, oauth2-proxy, etc.). This is the trusted source.
//
// Priority 2: metadata["user_id"] from the Response API request body.
// This is untrusted (client-provided) and intended for development/testing only.
func extractUserID(ctx *RequestContext) string {
	// Check auth header first (trusted source, injected by auth backend)
	if userID, ok := ctx.Headers[headers.AuthzUserID]; ok && userID != "" {
		logging.Debugf("Memory: Using user_id from auth header (%s)", headers.AuthzUserID)
		return userID
	}

	// DEV-ONLY: Fallback to metadata["user_id"] (untrusted, for development/testing)
	if ctx.ResponseAPICtx != nil && ctx.ResponseAPICtx.OriginalRequest != nil {
		if ctx.ResponseAPICtx.OriginalRequest.Metadata != nil {
			if userID, ok := ctx.ResponseAPICtx.OriginalRequest.Metadata["user_id"]; ok && userID != "" {
				logging.Warnf("Memory: Using user_id from request metadata (DEV BUILD - UNTRUSTED fallback)")
				return userID
			}
		}
	}

	return ""
}
