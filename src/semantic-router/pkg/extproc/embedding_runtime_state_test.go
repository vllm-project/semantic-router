package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// A remote default embedding deployment that only the response cache recalls
// with is prepared in the cache's set, and the startup status reports it from
// there; a deployment no consumer prepared is not reported.
func TestEmbeddingRuntimeStateReportsTheProviderARecallStorePrepared(t *testing.T) {
	remote, err := embedding.NewFuncProvider(config.EmbeddingBackendOpenAICompatible, 4, func(context.Context, string) ([]float32, error) {
		return []float32{1, 0, 0, 0}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.EmbeddingDeployment = "remote"
	router := &OpenAIRouter{Config: cfg, Embeddings: embedding.NewSet(nil, "remote")}
	if state := router.embeddingRuntimeState(); state.EmbeddingProvider != nil {
		t.Fatalf("no consumer prepared the deployment, but the status reports %+v", state.EmbeddingProvider)
	}

	router.cacheEmbeddings = embedding.NewSet(map[string]embedding.Provider{"remote": remote}, "remote")
	state := router.embeddingRuntimeState()
	if state.EmbeddingProvider == nil || state.EmbeddingProvider.Dimension != 4 || !state.AnyReady {
		t.Fatalf("the response cache's remote provider is not reported: %+v", state.EmbeddingProvider)
	}
}
