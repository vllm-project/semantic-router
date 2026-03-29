package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestCacheScopeUserID_PrefersAuthHeaderOverFallback(t *testing.T) {
	t.Setenv("SEMANTIC_CACHE_FALLBACK_USER_HEADER", "x-vsr-e2e-cache-user")
	ctx := &RequestContext{
		Headers: map[string]string{
			headers.AuthzUserID:    "auth-user",
			"x-vsr-e2e-cache-user": "other",
		},
	}
	assert.Equal(t, "auth-user", cacheScopeUserID(ctx))
}

func TestCacheScopeUserID_UsesFallbackWhenAuthMissing(t *testing.T) {
	t.Setenv("SEMANTIC_CACHE_FALLBACK_USER_HEADER", "x-vsr-e2e-cache-user")
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-vsr-e2e-cache-user": "fallback-user",
		},
	}
	assert.Equal(t, "fallback-user", cacheScopeUserID(ctx))
}

func TestCacheScopeUserID_UsesOpenAIUserFieldWhenEnvBody(t *testing.T) {
	t.Setenv("SEMANTIC_CACHE_E2E_USER_FROM_BODY", "true")
	ctx := &RequestContext{
		Headers:             map[string]string{},
		OriginalRequestBody: []byte(`{"model":"MoM","messages":[{"role":"user","content":"hi"}],"user":"body-user"}`),
	}
	assert.Equal(t, "body-user", cacheScopeUserID(ctx))
}

func TestCacheScopeUserID_AuthHeaderWinsOverBody(t *testing.T) {
	t.Setenv("SEMANTIC_CACHE_E2E_USER_FROM_BODY", "true")
	ctx := &RequestContext{
		Headers: map[string]string{
			headers.AuthzUserID: "hdr-user",
		},
		OriginalRequestBody: []byte(`{"user":"body-user"}`),
	}
	assert.Equal(t, "hdr-user", cacheScopeUserID(ctx))
}

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

func TestExtractUserID_AuthHeaderExactKey(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			headers.AuthzUserID: "user_exact_key",
		},
	}

	result := extractUserID(ctx)
	assert.Equal(t, "user_exact_key", result)
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
