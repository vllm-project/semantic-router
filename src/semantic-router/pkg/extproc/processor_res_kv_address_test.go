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
			principalA := testKVAddressRoutingContext(t)
			principalA.RequestID = "req-a"
			principalA.SessionID = sharedSession
			principalA.SessionProvenance = SessionProvenanceHeader
			principalA.RequestModel = "qwen3-14b"
			principalA.UpstreamBackendAddress = "10.0.1.5:8000"
			principalA.UpstreamStatusCode = 200
			principalA.AuthenticatedPrincipal = "user-a"
			principalA.Headers[headers.AuthzUserID] = "user-a"
			principalA.VSRSelectedDecision = responseCacheScopedDecision(scope)

			principalB := testKVAddressRoutingContext(t)
			principalB.RequestID = "req-b"
			principalB.SessionID = sharedSession
			principalB.SessionProvenance = SessionProvenanceHeader
			principalB.RequestModel = "qwen3-14b"
			principalB.UpstreamBackendAddress = "10.0.1.6:8000"
			principalB.UpstreamStatusCode = 200
			principalB.AuthenticatedPrincipal = "user-b"
			principalB.Headers[headers.AuthzUserID] = "user-b"
			principalB.Headers[headers.AuthzTenantID] = "tenant-shared"
			principalB.Headers[headers.AuthzTeamID] = "team-shared"
			principalB.VSRSelectedDecision = responseCacheScopedDecision(scope)

			router.updateKVAddressRegistry(principalA)
			router.updateKVAddressRegistry(principalB)

			if kvAddressNamespace(principalA) == kvAddressNamespace(principalB) {
				t.Fatalf("principals share namespace %q under response-cache scope %q", kvAddressNamespace(principalA), scope)
			}

			gotA, err := reg.Lookup(context.Background(), kvAddressNamespace(principalA), routingSessionStateKey(principalA))
			if err != nil {
				t.Fatalf("Lookup(principalA) error = %v", err)
			}
			gotB, err := reg.Lookup(context.Background(), kvAddressNamespace(principalB), routingSessionStateKey(principalB))
			if err != nil {
				t.Fatalf("Lookup(principalB) error = %v", err)
			}
			if gotA == nil || gotA.SourcePod != "10.0.1.5:8000" {
				t.Fatalf("principalA SourcePod = %v, want 10.0.1.5:8000", gotA)
			}
			if gotB == nil || gotB.SourcePod != "10.0.1.6:8000" {
				t.Fatalf("principalB SourcePod = %v, want 10.0.1.6:8000", gotB)
			}
		})
	}
}

type closeTrackingAddressRegistry struct {
	closeCalls int
}

func (*closeTrackingAddressRegistry) Write(context.Context, kvtransfer.AddressRecord) error {
	return nil
}

func (*closeTrackingAddressRegistry) Lookup(context.Context, string, string) (*kvtransfer.AddressRecord, error) {
	return nil, nil
}

func (r *closeTrackingAddressRegistry) Close() error {
	r.closeCalls++
	return nil
}

func TestOpenAIRouterCloseClosesKVAddressRegistry(t *testing.T) {
	reg := &closeTrackingAddressRegistry{}
	resources := newResourceScope()
	router := &OpenAIRouter{resources: resources}

	router.kvAddressRegistryMu.Lock()
	router.setKVAddressRegistryLocked(reg)
	router.kvAddressRegistryMu.Unlock()

	if reg.closeCalls != 0 {
		t.Fatalf("AddressRegistry.Close() called %d times before router.Close(), want 0", reg.closeCalls)
	}
	if err := router.Close(); err != nil {
		t.Fatalf("router.Close() error = %v", err)
	}
	if reg.closeCalls != 1 {
		t.Fatalf("AddressRegistry.Close() called %d times after router.Close(), want 1", reg.closeCalls)
	}

	if err := router.Close(); err != nil {
		t.Fatalf("second router.Close() error = %v", err)
	}
	if reg.closeCalls != 1 {
		t.Fatalf("AddressRegistry.Close() called %d times after second Close, want 1", reg.closeCalls)
	}
}

func TestRouterGenerationCloseReleasesKVAddressRegistry(t *testing.T) {
	reg := &closeTrackingAddressRegistry{}
	router := (&routerComponents{resources: newResourceScope()}).buildRouter()
	router.kvAddressRegistryMu.Lock()
	router.setKVAddressRegistryLocked(reg)
	router.kvAddressRegistryMu.Unlock()

	service := NewRouterService(router)
	if err := service.Close(); err != nil {
		t.Fatalf("RouterService.Close() error = %v", err)
	}
	if reg.closeCalls != 1 {
		t.Fatalf("AddressRegistry.Close() called %d times on generation teardown, want 1", reg.closeCalls)
	}
}
