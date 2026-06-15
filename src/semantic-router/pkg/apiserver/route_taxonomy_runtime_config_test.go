//go:build !windows && cgo

package apiserver

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestHandleKnowledgeBaseMutationWithRuntimeRegistryDoesNotReplaceGlobalConfig(t *testing.T) {
	apiServer, _, _ := newTestKnowledgeBaseAPIServer(t)
	apiServer.runtimeRegistry = routerruntime.NewRegistry(apiServer.config)
	withStubbedRuntimeConfigSync(t)

	globalCfg := &config.RouterConfig{ConfigSource: config.ConfigSourceKubernetes}
	restoreGlobalConfig := replaceGlobalConfigForTest(globalCfg)
	t.Cleanup(restoreGlobalConfig)

	created := createKnowledgeBaseDocument(t, apiServer, testKnowledgeBasePayload())
	if created.Name != "research_kb" {
		t.Fatalf("expected research_kb to be created, got %+v", created)
	}

	if got := config.Get(); got != globalCfg {
		t.Fatalf("config.Get() = %p, want unchanged global config %p", got, globalCfg)
	}
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
