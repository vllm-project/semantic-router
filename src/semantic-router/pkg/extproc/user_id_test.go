package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestIdentityScopeUsesAuthenticatedTenantContext(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "spoofed-user",
			"x-authz-team-id": "spoofed-team",
		},
		InferenceAccess: &inferenceRequestAccess{tenant: accessruntime.TenantContext{
			NamespaceID: "namespace-1",
			UserID:      "user-1",
			TeamID:      "team-1",
		}},
	}
	if got := cacheScopeUserID(ctx); got != "user-1" {
		t.Fatalf("cache user = %q, want authenticated user", got)
	}
	if got := extractUserID(ctx); got != "user-1" {
		t.Fatalf("memory user = %q, want authenticated user", got)
	}

	payload, err := config.NewStructuredPayload(config.ResponseCachePluginConfig{Enabled: true, Scope: "team"})
	if err != nil {
		t.Fatal(err)
	}
	ctx.VSRSelectedDecision = &config.Decision{Plugins: []config.DecisionPlugin{{
		Type: "response_cache", Configuration: payload,
	}}}
	if got := responseCacheScopeIdentity(ctx); got != "team-1" {
		t.Fatalf("team cache scope = %q", got)
	}
}

func TestIdentityScopeIgnoresRequestHeadersAndBodyIdentity(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-authz-user-id": "spoofed-user",
		},
		SemanticRequest: &llmprotocol.Request{Generation: 1, Metadata: map[string]string{
			"user_id": "metadata-user",
		}},
	}
	if got := cacheScopeUserID(ctx); got != "" {
		t.Fatalf("untrusted cache identity = %q", got)
	}
	if got := extractUserID(ctx); got != "" {
		t.Fatalf("untrusted memory identity = %q", got)
	}
}

func TestRouterLearningSessionIsPartitionedByAuthenticatedTenant(t *testing.T) {
	cfg := config.RouterLearningProtectionConfig{Scope: config.RouterLearningScopeSession}
	ctx := &RequestContext{
		Headers: map[string]string{"x-session-id": "shared-session"},
		InferenceAccess: &inferenceRequestAccess{tenant: accessruntime.TenantContext{
			NamespaceID: "namespace-1",
			APIKeyID:    "key-1",
		}},
	}
	identity, ok := (&OpenAIRouter{}).protectionIdentity(ctx, cfg)
	if !ok {
		t.Fatal("authenticated learning identity was rejected")
	}
	if identity.memoryKey != "namespace-1/key-1/shared-session" {
		t.Fatalf("learning memory key = %q", identity.memoryKey)
	}

	ctx.InferenceAccess.tenant.APIKeyID = ""
	if _, ok := (&OpenAIRouter{}).protectionIdentity(ctx, cfg); ok {
		t.Fatal("authenticated learning identity accepted incomplete TenantContext")
	}
}
