package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/kvtransfer"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestKVConfiguredSourcePublicationAndTargetHandoff(t *testing.T) {
	t.Setenv("USER_SCOPE_NAMESPACE_SECRET", "test-secret")
	source := config.KVServingIdentity{Model: "source", WeightRevision: strings.Repeat("a", 40), Tokenizer: "shared", TokenizerRevision: strings.Repeat("b", 40), Precision: "bf16", TensorParallel: 1, KVHeads: 8, HeadDim: 128, HeadOrder: "contiguous"}
	target := source
	target.Model = "target"
	target.WeightRevision = strings.Repeat("c", 40)
	registry := kvtransfer.NewMemoryAddressRegistry()
	router := &OpenAIRouter{kvAddressRegistryStore: registry, Config: &config.RouterConfig{KVTransfer: &config.KVTransferConfig{
		Enabled:  true,
		Backends: []config.KVTransferBackend{{ModelAlias: "source-alias", BackendName: "source-backend", Identity: source, CanExport: true}, {ModelAlias: "target-alias", BackendName: "target-backend", Identity: target, CanLoad: true}},
		Pairs:    []config.KVTransferPair{{MapperID: "mapper", Source: source, Target: target, Enabled: true, MaxTransferTurn: 3}},
	}}}
	router.initializeKVHandoff()
	ctx := testKVAddressRoutingContext(t)
	ctx.AuthenticatedPrincipal = "tenant-a"
	ctx.SessionProvenance = SessionProvenanceHeader
	ctx.SessionID = "session"
	ctx.RequestID = "source-request"
	ctx.RequestModel = "source-alias"
	ctx.primaryBackendName = "source-backend"
	body, err := router.encodeKVHandoff([]byte(`{"model":"source"}`), llmprotocol.OpenAIChatV1, ctx)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if decodeErr := json.Unmarshal(body, &wire); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if wire["kv_transfer_params"] == nil || ctx.KVSourceIdentity == nil {
		t.Fatal("source export was not requested")
	}
	ctx.UpstreamBackendAddress = "10.0.0.1:8000"
	ctx.UpstreamStatusCode = 200
	router.updateKVAddressRegistry(ctx)
	record, err := registry.Lookup(context.Background(), cache.UserScopeNamespace("tenant-a"), routingSessionStateKey(ctx))
	if err != nil || record == nil || record.CacheID != ctx.KVSourceCacheID || !record.ExpiresAt.After(time.Now()) {
		t.Fatalf("record = %v, %v", record, err)
	}
	hint, reason := router.KVHandoff.PlanDispatch(context.Background(), KVDispatch{Principal: "tenant-a", SessionID: routingSessionStateKey(ctx), SessionProvenance: "header", PreviousModel: "source-alias", TargetModel: "target-alias", BackendName: "target-backend", Turn: 1})
	if hint == nil || reason != kvtransfer.ReasonEligible || hint.CacheID != ctx.KVSourceCacheID {
		t.Fatalf("handoff = %v, %s", hint, reason)
	}
	hint, _ = router.KVHandoff.PlanDispatch(context.Background(), KVDispatch{Principal: "tenant-b", SessionID: routingSessionStateKey(ctx), SessionProvenance: "header", TargetModel: "target-alias", BackendName: "target-backend", Turn: 1})
	if hint != nil {
		t.Fatal("another tenant reused the source")
	}
	for _, backend := range []config.KVTransferBackend{
		{ModelAlias: "source-alias", BackendName: "source-backend", Identity: source, CanExport: false},
		{ModelAlias: "source-alias", BackendName: "source-backend", Identity: target, CanExport: true},
	} {
		router.Config.KVTransfer.Backends[0] = backend
		hint, _ = router.KVHandoff.PlanDispatch(context.Background(), KVDispatch{Principal: "tenant-a", SessionID: routingSessionStateKey(ctx), SessionProvenance: "header", TargetModel: "target-alias", BackendName: "target-backend", Turn: 1})
		if hint != nil {
			t.Fatal("stale deployment identity or disabled export reused the source")
		}
	}
}
