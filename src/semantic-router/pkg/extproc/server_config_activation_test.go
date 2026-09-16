package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestReloadFailurePublishesActivationCauseWithoutReplacingRuntime(t *testing.T) {
	defer stubReloadSeams(t)()
	previous := &OpenAIRouter{Config: &config.RouterConfig{DocumentHash: "active"}}
	registry := routerruntime.NewRegistry(previous.Config)
	server := &Server{service: NewRouterService(previous), runtime: registry}
	prepareReloadRuntime = func(*config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
		return modelruntime.EmbeddingRuntimeState{}, errors.New("test dependency unavailable")
	}
	candidate := &config.RouterConfig{DocumentHash: "candidate"}
	if err := server.reloadRouterFromConfig("kubernetes", "config.yaml", candidate); err == nil {
		t.Fatal("expected failed activation")
	}
	state := registry.ConfigActivation()
	if state.Status != "failed" || state.Stage != "dependencies" || state.DocumentHash != candidate.DocumentHash || state.FailureDetail == "" {
		t.Fatalf("failed reload not observable: %+v", state)
	}
	if registry.CurrentConfig() != previous.Config || server.service.GetRouter() != previous {
		t.Fatal("failed reload replaced active generation")
	}
}
