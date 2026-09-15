package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/kvtransfer"
)

func testKVAddressRoutingContext(t *testing.T) *RequestContext {
	t.Helper()
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
	return ctx
}

func TestShouldWriteKVAddressRegistryRequiresSuccessfulUpstream(t *testing.T) {
	ctx := testKVAddressRoutingContext(t)
	ctx.SessionID = "sess-1"
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
	ctx.RequestModel = "qwen3-14b"

	if shouldWriteKVAddressRegistry(ctx) {
		t.Fatal("shouldWriteKVAddressRegistry() = true, want false without backend address")
	}
}

func TestUpdateKVAddressRegistryWritesSuccessfulTurn(t *testing.T) {
	reg := kvtransfer.NewMemoryAddressRegistry()
	router := &OpenAIRouter{kvAddressRegistryStore: reg}
	ctx := testKVAddressRoutingContext(t)
	ctx.RequestID = "req-1"
	ctx.SessionID = "sess-abc"
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
}
