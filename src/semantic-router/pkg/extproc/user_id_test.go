package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// =============================================================================
// extractUserID Tests (Common to both dev and prod builds)
// =============================================================================

func TestExtractUserID_AuthHeaderOnly(t *testing.T) {
	// Auth header present, no metadata
	ctx := &RequestContext{
		Headers: map[string]string{
			headers.AuthzUserID: "user_from_auth",
		},
	}

	result := extractUserID(ctx)
	assert.Equal(t, "user_from_auth", result)
}

func TestExtractUserID_NoAuthHeaderNoMetadata(t *testing.T) {
	// Neither auth header nor metadata present
	ctx := &RequestContext{
		Headers: map[string]string{},
	}

	result := extractUserID(ctx)
	assert.Empty(t, result, "should return empty string when no user ID source available")
}

func TestExtractUserID_UnrelatedHeaderIgnored(t *testing.T) {
	// Unrelated headers should not be used as user ID
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-custom-user-id": "user_from_wrong_header",
			"authorization":    "Bearer token123",
		},
	}

	result := extractUserID(ctx)
	assert.Empty(t, result, "should not use unrelated headers as user ID")
}
