package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func TestRouterVectorStoreRuntimeUsesRuntimeRegistryOnly(t *testing.T) {
	registry := routerruntime.NewRegistry(nil)
	router := &OpenAIRouter{RuntimeRegistry: registry}

	if got := router.currentVectorStoreManager(); got != nil {
		t.Fatalf("currentVectorStoreManager() = %v, want nil before runtime publication", got)
	}
	if got := router.currentVectorStoreEmbedder(); got != nil {
		t.Fatalf("currentVectorStoreEmbedder() = %v, want nil before runtime publication", got)
	}

	manager := &vectorstore.Manager{}
	embedder := runtimeServicesEmbedder{}
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{
		Manager:  manager,
		Embedder: embedder,
	})

	if got := router.currentVectorStoreManager(); got != manager {
		t.Fatalf("currentVectorStoreManager() = %v, want runtime manager %v", got, manager)
	}
	if got := router.currentVectorStoreEmbedder(); got != embedder {
		t.Fatalf("currentVectorStoreEmbedder() = %v, want runtime embedder %v", got, embedder)
	}
}

func TestRouterVectorStoreRuntimeNilRouterAndNilRegistry(t *testing.T) {
	var router *OpenAIRouter
	if got := router.currentVectorStoreManager(); got != nil {
		t.Fatalf("currentVectorStoreManager() = %v, want nil for nil router", got)
	}
	if got := router.currentVectorStoreEmbedder(); got != nil {
		t.Fatalf("currentVectorStoreEmbedder() = %v, want nil for nil router", got)
	}

	router = &OpenAIRouter{}
	if got := router.currentVectorStoreManager(); got != nil {
		t.Fatalf("currentVectorStoreManager() = %v, want nil without runtime registry", got)
	}
	if got := router.currentVectorStoreEmbedder(); got != nil {
		t.Fatalf("currentVectorStoreEmbedder() = %v, want nil without runtime registry", got)
	}
}

type runtimeServicesEmbedder struct{}

func (runtimeServicesEmbedder) Embed(_ string) ([]float32, error) {
	return []float32{1, 0}, nil
}

func (runtimeServicesEmbedder) Dimension() int {
	return 2
}
