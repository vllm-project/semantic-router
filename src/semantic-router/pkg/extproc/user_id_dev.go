//go:build dev

package extproc

import (
	"encoding/json"

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

	// DEV-ONLY: Fallback to Chat Completions metadata["user_id"] or "user" field (untrusted)
	if len(ctx.ChatCompletionRequestBody) > 0 {
		// Extract on-demand in dev builds only
		if ctx.ChatCompletionUserID == "" {
			ctx.ChatCompletionUserID = extractChatCompletionUserIDFromBody(ctx.ChatCompletionRequestBody)
		}
		if ctx.ChatCompletionUserID != "" {
			logging.Warnf("Memory: Using user_id from Chat Completions (DEV BUILD - UNTRUSTED fallback)")
			return ctx.ChatCompletionUserID
		}
	}

	return ""
}

// extractChatCompletionUserIDFromBody extracts user ID from Chat Completions request body.
// Priority: metadata["user_id"] > "user" field (deprecated).
//
// DEV BUILD ONLY: This function only exists in dev builds and extracts UNTRUSTED
// client-provided data as a fallback for development/testing.
func extractChatCompletionUserIDFromBody(requestBody []byte) string {
	var req struct {
		Metadata map[string]string `json:"metadata"`
		User     string            `json:"user"`
	}
	if err := json.Unmarshal(requestBody, &req); err != nil {
		return ""
	}

	// Priority 1: metadata["user_id"] (consistent with Response API)
	if req.Metadata != nil {
		if userID, ok := req.Metadata["user_id"]; ok && userID != "" {
			return userID
		}
	}

	// Priority 2: "user" field (deprecated by OpenAI, for backward compatibility)
	return req.User
}
