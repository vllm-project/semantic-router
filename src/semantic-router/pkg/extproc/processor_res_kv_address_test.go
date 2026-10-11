package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/kvtransfer"
)

func testKVAddressRoutingContext(t *testing.T) *RequestContext {
	t.Helper()
	ctx := &RequestContext{
		AuthenticatedPrincipal: "user-a",
		Headers: map[string]string{
			headers.AuthzUserID:   "user-a",
			headers.AuthzTenantID: "tenant-shared",
			headers.AuthzTeamID:   "team-shared",
		},
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	return ctx
}

func responseCacheScopedDecision(scope string) *config.Decision {
	return &config.Decision{
		Name: "route",
		Plugins: []config.DecisionPlugin{{
			Type:          config.DecisionPluginResponseCache,
			Configuration: config.MustStructuredPayload(config.ResponseCachePluginConfig{Enabled: true, Scope: scope}),
		}},
	}
}

func TestShouldWriteKVAddressRegistryRequiresSuccessfulUpstream(t *testing.T) {
	ctx := testKVAddressRoutingContext(t)
	ctx.SessionID = "sess-1"
	ctx.SessionProvenance = SessionProvenanceHeader
	ctx.RequestModel = "qwen3-14b"
	ctx.UpstreamBackendAddress = "10.0.1.5:8000"
	ctx.UpstreamStatusCode = 503

	if shouldWriteKVAddressRegistry(ctx) {
		t.Fatal("shouldWriteKVAddressRegistry() = true, want false for non-2xx upstream")
	}
}

func TestShouldWriteKVAddressRegistryRequiresBackendAddress(t *testing.T) {
	ctx := testKVAddressRoutingContext(t)
	ctx.SessionID = "sess-1"
	ctx.SessionProvenance = SessionProvenanceHeader
	ctx.RequestModel = "qwen3-14b"

	if shouldWriteKVAddressRegistry(ctx) {
		t.Fatal("shouldWriteKVAddressRegistry() = true, want false without backend address")
	}
}

func TestShouldWriteKVAddressRegistryRequiresAuthenticatedPrincipal(t *testing.T) {
	ctx := testKVAddressRoutingContext(t)
	ctx.SessionID = "sess-1"
	ctx.SessionProvenance = SessionProvenanceHeader
	ctx.RequestModel = "qwen3-14b"
	ctx.UpstreamBackendAddress = "10.0.1.5:8000"
	ctx.UpstreamStatusCode = 200
	ctx.AuthenticatedPrincipal = ""

	if shouldWriteKVAddressRegistry(ctx) {
		t.Fatal("shouldWriteKVAddressRegistry() = true, want false without authenticated principal")
	}
}

func TestShouldWriteKVAddressRegistrySkipsDerivedSessionProvenance(t *testing.T) {
	ctx := testKVAddressRoutingContext(t)
	ctx.SessionID = "cc-full-a1b2c3d4e5f67890"
	ctx.SessionProvenance = SessionProvenanceMessageHash
	ctx.RequestModel = "qwen3-14b"
	ctx.UpstreamBackendAddress = "10.0.1.5:8000"
	ctx.UpstreamStatusCode = 200

	if shouldWriteKVAddressRegistry(ctx) {
		t.Fatal("shouldWriteKVAddressRegistry() = true, want false for message_hash / cc-full sessions")
	}
}

func TestUpdateKVAddressRegistryWritesSuccessfulTurn(t *testing.T) {
	reg := kvtransfer.NewMemoryAddressRegistry()
	router := &OpenAIRouter{kvAddressRegistryStore: reg}
	ctx := testKVAddressRoutingContext(t)
	ctx.RequestID = "req-1"
	ctx.SessionID = "sess-abc"
	ctx.SessionProvenance = SessionProvenanceHeader
	ctx.RequestModel = "qwen3-14b"
	ctx.TurnIndex = 4
	ctx.UpstreamBackendAddress = "10.0.1.5:8000"
	ctx.UpstreamStatusCode = 200

	router.updateKVAddressRegistry(ctx)

	sessionKey := routingSessionStateKey(ctx)
	namespace := kvAddressNamespace(ctx)
	got, err := reg.Lookup(context.Background(), namespace, sessionKey)
	if err != nil {
		t.Fatalf("Lookup() error = %v", err)
	}
	if got == nil {
		t.Fatal("Lookup() = nil, want KV address record")
	}
	if got.SourcePod != "10.0.1.5:8000" {
		t.Fatalf("SourcePod = %q, want %q", got.SourcePod, "10.0.1.5:8000")
	}
	if got.Model != "qwen3-14b" {
		t.Fatalf("Model = %q, want %q", got.Model, "qwen3-14b")
	}
	if got.TurnCount != 5 {
		t.Fatalf("TurnCount = %d, want 5", got.TurnCount)
	}
	if got.SessionID != sessionKey {
		t.Fatalf("SessionID = %q, want %q", got.SessionID, sessionKey)
	}
	if namespace != cache.UserScopeNamespace("user-a") {
		t.Fatalf("Namespace = %q, want principal-isolated namespace", namespace)
	}
}

func TestKVAddressNamespaceIsolatesPrincipalsAcrossResponseCacheScopes(t *testing.T) {
	reg := kvtransfer.NewMemoryAddressRegistry()
	router := &OpenAIRouter{kvAddressRegistryStore: reg}
	const sharedSession = "sess-shared"

	for _, scope := range []string{"", "global", "tenant", "team", "user"} {
		t.Run("scope_"+scope, func(t *testing.T) {
			userA := testKVAddressRoutingContext(t)
			userA.RequestID = "req-a"
			userA.SessionID = sharedSession
			userA.SessionProvenance = SessionProvenanceHeader
			userA.RequestModel = "qwen3-14b"
			userA.UpstreamBackendAddress = "10.0.1.5:8000"
			userA.UpstreamStatusCode = 200
			userA.AuthenticatedPrincipal = "user-a"
			userA.Headers[headers.AuthzUserID] = "user-a"
			userA.VSRSelectedDecision = responseCacheScopedDecision(scope)

			userB := testKVAddressRoutingContext(t)
			userB.RequestID = "req-b"
			userB.SessionID = sharedSession
			userB.SessionProvenance = SessionProvenanceHeader
			userB.RequestModel = "qwen3-14b"
			userB.UpstreamBackendAddress = "10.0.1.6:8000"
			userB.UpstreamStatusCode = 200
			userB.AuthenticatedPrincipal = "user-b"
			userB.Headers[headers.AuthzUserID] = "user-b"
			userB.Headers[headers.AuthzTenantID] = "tenant-shared"
			userB.Headers[headers.AuthzTeamID] = "team-shared"
			userB.VSRSelectedDecision = responseCacheScopedDecision(scope)

			router.updateKVAddressRegistry(userA)
			router.updateKVAddressRegistry(userB)

			if kvAddressNamespace(userA) == kvAddressNamespace(userB) {
				t.Fatalf("principals share namespace %q under response-cache scope %q", kvAddressNamespace(userA), scope)
			}

			gotA, err := reg.Lookup(context.Background(), kvAddressNamespace(userA), routingSessionStateKey(userA))
			if err != nil {
				t.Fatalf("Lookup(userA) error = %v", err)
			}
			gotB, err := reg.Lookup(context.Background(), kvAddressNamespace(userB), routingSessionStateKey(userB))
			if err != nil {
				t.Fatalf("Lookup(userB) error = %v", err)
			}
			if gotA == nil || gotA.SourcePod != "10.0.1.5:8000" {
				t.Fatalf("userA SourcePod = %v, want 10.0.1.5:8000", gotA)
			}
			if gotB == nil || gotB.SourcePod != "10.0.1.6:8000" {
				t.Fatalf("userB SourcePod = %v, want 10.0.1.6:8000", gotB)
			}
		})
	}
}
