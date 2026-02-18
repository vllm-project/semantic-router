//go:build dev

package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// =============================================================================
// extractUserID Tests (Dev build only - tests metadata fallback)
// =============================================================================

func TestExtractUserID_AuthHeaderTakesPrecedence(t *testing.T) {
	// Auth header (x-authz-user-id) takes precedence over metadata["user_id"]
	ctx := &RequestContext{
		Headers: map[string]string{
			headers.AuthzUserID: "user_from_auth",
		},
		ResponseAPICtx: &ResponseAPIContext{
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Metadata: map[string]string{
					"user_id": "user_from_metadata",
				},
			},
		},
	}

	result := extractUserID(ctx)
	assert.Equal(t, "user_from_auth", result, "auth header should take precedence over metadata")
}

func TestExtractUserID_FallbackToMetadataWhenNoAuthHeader(t *testing.T) {
	// No auth header, falls back to metadata["user_id"]
	ctx := &RequestContext{
		Headers: map[string]string{},
		ResponseAPICtx: &ResponseAPIContext{
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Metadata: map[string]string{
					"user_id": "user_from_metadata",
				},
			},
		},
	}

	result := extractUserID(ctx)
	assert.Equal(t, "user_from_metadata", result, "should fall back to metadata when auth header absent")
}

func TestExtractUserID_EmptyAuthHeaderFallsBackToMetadata(t *testing.T) {
	// Auth header present but empty, falls back to metadata
	ctx := &RequestContext{
		Headers: map[string]string{
			headers.AuthzUserID: "",
		},
		ResponseAPICtx: &ResponseAPIContext{
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Metadata: map[string]string{
					"user_id": "user_from_metadata",
				},
			},
		},
	}

	result := extractUserID(ctx)
	assert.Equal(t, "user_from_metadata", result, "empty auth header should fall back to metadata")
}

// =============================================================================
// extractMemoryInfo Tests (Dev build only - tests metadata fallback path)
// =============================================================================

func TestExtractMemoryInfo_FallbackToMetadata(t *testing.T) {
	// No auth header, falls back to metadata["user_id"] (dev-only behavior)
	ctx := &RequestContext{
		RequestID: "req_123",
		Headers:   map[string]string{},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv_from_translate",
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Metadata: map[string]string{
					"user_id": "user_from_metadata",
				},
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when falling back to metadata")
	assert.Equal(t, "conv_from_translate", sessionID)
	assert.Equal(t, "user_from_metadata", userID, "should fall back to metadata when no auth header")
	assert.Empty(t, history)
}

func TestExtractMemoryInfo_UserIDFromMetadataOnly(t *testing.T) {
	// Tests that metadata["user_id"] works as a source in dev builds
	ctx := &RequestContext{
		RequestID: "req_123",
		Headers:   map[string]string{},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv_from_translate",
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Metadata: map[string]string{
					"user_id": "user_from_metadata",
				},
			},
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(ctx)

	require.NoError(t, err, "should not return error when userID is provided via metadata")
	assert.Equal(t, "conv_from_translate", sessionID)
	assert.Equal(t, "user_from_metadata", userID)
	assert.Empty(t, history)
}
