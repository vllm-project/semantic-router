//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestHandleKnowledgeBaseMutationWithRuntimeRegistryDoesNotReplaceGlobalConfig(t *testing.T) {
	apiServer, _, configPath := newTestKnowledgeBaseAPIServer(t)
	old := apiServer.config
	apiServer.runtimeRegistry = routerruntime.NewRegistry(apiServer.config)
	apiServer.runtimeConfig = newLiveRuntimeConfig(old, apiServer.runtimeRegistry.CurrentConfig, nil)
	withStubbedRuntimeConfigSync(t)

	globalCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	restoreGlobalConfig := replaceGlobalConfigForTest(globalCfg)
	t.Cleanup(restoreGlobalConfig)

	request := httptest.NewRequest(http.MethodPost, "/api/v1/storage/knowledge-bases", bytes.NewReader(mustMarshalKnowledgeBasePayload(t, testKnowledgeBasePayload())))
	response := httptest.NewRecorder()
	apiServer.handleCreateKnowledgeBase(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("expected pending candidate, got %d: %s", response.Code, response.Body.String())
	}
	created := mustDecodeKnowledgeBaseDocument(t, response)
	if created.Name != "research_kb" {
		t.Fatalf("expected research_kb to be created, got %+v", created)
	}

	if got := config.Get(); got != globalCfg {
		t.Fatalf("config.Get() = %p, want unchanged global config %p", got, globalCfg)
	}
	if apiServer.runtimeRegistry.CurrentConfig() != old || apiServer.currentConfig() != old {
		t.Fatal("KB write replaced the active generation before preparation")
	}
	if created.ActivationStatus != "pending" || created.GeneratedRuntimeHash == "" {
		t.Fatalf("missing candidate state: %+v", created)
	}
	// The file watcher prepares and publishes the whole candidate atomically.
	candidate, err := config.Parse(configPath)
	if err != nil {
		t.Fatal(err)
	}
	apiServer.runtimeRegistry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: candidate})
	runtimeCfg := apiServer.runtimeRegistry.CurrentConfig()
	if runtimeCfg == nil {
		t.Fatalf("runtime registry config was not updated")
	}
	if runtimeCfg == globalCfg {
		t.Fatalf("runtime registry adopted global config %p", globalCfg)
	}
	wantKbs := append(defaultKnowledgeBaseNames(), "research_kb")
	assertKnowledgeBaseConfigNames(t, runtimeCfg.KnowledgeBases, wantKbs)
	assertKnowledgeBaseConfigNames(t, apiServer.currentConfig().KnowledgeBases, wantKbs)
}
