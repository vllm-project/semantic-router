//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestReplayManagementPreservesProviderCacheEvidence(t *testing.T) {
	zero := 0
	memory := store.NewMemoryStore(10, 0)
	accounting := &store.ProviderCacheAccounting{State: "partial", CacheRead: store.TokenCostComponent{Tokens: &zero, Provenance: "authoritative"}, CacheWrite: store.TokenCostComponent{Provenance: "unknown"}}
	id, err := memory.Add(context.Background(), store.Record{ID: "cache-accounting", ProviderCache: accounting})
	if err != nil {
		t.Fatal(err)
	}
	record, _, err := memory.Get(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	runtime := &replayRuntimeStub{handled: true, response: routerruntime.ReplayResponse{StatusCode: http.StatusOK, Body: body}}
	registry := routerruntime.NewRegistry(cfg)
	registry.SetReplayRuntime(runtime)
	server := &ClassificationAPIServer{config: cfg, runtimeRegistry: registry}
	response := serveReplayManagementRequest(server.setupRoutes(), "/api/v1/observability/replays/"+id, "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var decoded store.Record
	if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.ProviderCache == nil || decoded.ProviderCache.CacheRead.Tokens == nil || *decoded.ProviderCache.CacheRead.Tokens != 0 {
		t.Fatal("management API lost authoritative zero")
	}
	if decoded.ProviderCache.CacheWrite.Tokens != nil || decoded.ProviderCache.CacheSavings != nil {
		t.Fatal("management API invented missing accounting")
	}
}
