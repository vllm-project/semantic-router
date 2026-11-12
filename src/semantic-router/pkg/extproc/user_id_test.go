package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
		Headers: map[string]string{},
		SemanticRequest: &llmprotocol.Request{Metadata: map[string]string{
			"user_id": "body-user",
		}},
	}
	assert.Equal(t, "body-user", cacheScopeUserID(ctx))
}

func TestCacheScopeUserID_AuthHeaderWinsOverBody(t *testing.T) {
	t.Setenv("SEMANTIC_CACHE_E2E_USER_FROM_BODY", "true")
	ctx := &RequestContext{
		Headers: map[string]string{
			headers.AuthzUserID: "hdr-user",
		},
		SemanticRequest: &llmprotocol.Request{Metadata: map[string]string{
			"user_id": "body-user",
		}},
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

// The response cache's user and team scopes and memory read the identity
// headers global.services.authz.identity names, as rate limits and the authz
// signal do; the default x-authz-* names then carry no identity.
func TestCacheScopeAndMemoryReadTheConfiguredIdentityHeaders(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		Authz: config.AuthzConfig{Identity: config.IdentityConfig{UserIDHeader: "x-jwt-sub", UserGroupsHeader: "x-jwt-groups"}},
	}}
	ctx := &RequestContext{Headers: make(map[string]string)}
	request := newDebugGateRequestHeaders(map[string]string{
		"x-jwt-sub": "alice", "x-jwt-groups": "research", headers.AuthzUserID: "bob", headers.AuthzUserGroups: "billing",
	})
	if _, err := router.handleRequestHeaders(request, ctx); err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	assert.Equal(t, "alice", extractUserID(ctx), "memory user")
	for scope, want := range map[string]string{"user": "alice", "team": "research"} {
		ctx.VSRSelectedDecision = &config.Decision{Name: "scoped", Plugins: []config.DecisionPlugin{{
			Type:          config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true, "scope": scope}),
		}}}
		assert.Equal(t, want, responseCacheScopeIdentity(ctx), "%s cache scope", scope)
	}
}
