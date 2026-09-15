package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestCacheScopeUserIDUsesTrustedIdentity(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id":      "spoofed-header-user",
			"x-vsr-e2e-cache-user": "other",
		},
		TrustedIdentity: authz.TrustedIdentity{UserID: "trusted-user"},
	}
	assert.Equal(t, "trusted-user", cacheScopeUserID(ctx))
}

func TestCacheScopeUserIDDoesNotUseFallbackHeader(t *testing.T) {
	t.Setenv("SEMANTIC_CACHE_FALLBACK_USER_HEADER", "x-vsr-e2e-cache-user")
	ctx := &RequestContext{Headers: map[string]string{
		"x-vsr-e2e-cache-user": "fallback-user",
	}}
	assert.Empty(t, cacheScopeUserID(ctx))
}

func TestCacheScopeUserIDDoesNotUseBodyMetadata(t *testing.T) {
	t.Setenv("SEMANTIC_CACHE_E2E_USER_FROM_BODY", "true")
	ctx := &RequestContext{
		Headers: map[string]string{},
		SemanticRequest: &llmprotocol.Request{Metadata: map[string]string{
			"user_id": "body-user",
		}},
	}
	assert.Empty(t, cacheScopeUserID(ctx))
}

func TestCacheScopeUserIDDoesNotUseRawHeaderWhenTypedIdentityMissing(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		"x-authz-user-id": "header-user",
	}}
	assert.Empty(t, cacheScopeUserID(ctx))
}

func TestExtractUserIDUsesTypedIdentity(t *testing.T) {
	ctx := &RequestContext{
		Headers:         map[string]string{"x-authz-user-id": "spoofed"},
		TrustedIdentity: authz.TrustedIdentity{UserID: "user_from_ingress"},
	}
	assert.Equal(t, "user_from_ingress", extractUserID(ctx))
}

func TestExtractUserIDDoesNotUseRawHeaderOrMetadata(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id":  "user_from_header",
			"x-custom-user-id": "user_from_wrong_header",
			"authorization":    "Bearer token123",
		},
		SemanticRequest: &llmprotocol.Request{Metadata: map[string]string{
			"user_id": "user_from_metadata",
		}},
	}
	assert.Empty(t, extractUserID(ctx))
}
