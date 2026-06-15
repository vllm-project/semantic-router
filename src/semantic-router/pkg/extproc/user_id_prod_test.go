//go:build !dev

package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// =============================================================================
// extractUserID Tests (Production build - no metadata fallback)
// =============================================================================

func TestExtractUserID_NoFallbackToMetadataInProduction(t *testing.T) {
	// In production, metadata["user_id"] should be ignored
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
	assert.Empty(t, result, "production build should not fall back to metadata")
}

func TestExtractUserID_EmptyAuthHeaderNoFallbackInProduction(t *testing.T) {
	// In production, empty auth header should NOT fall back to metadata
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
	assert.Empty(t, result, "production build should not fall back to metadata even with empty auth header")
}
