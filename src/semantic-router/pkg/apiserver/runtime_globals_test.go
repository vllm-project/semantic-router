//go:build !windows && cgo

package apiserver

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func TestLegacyRuntimeGlobalsResolveThroughSynchronizedAccessors(t *testing.T) {
	manager := &vectorstore.Manager{}
	pipeline := &vectorstore.IngestionPipeline{}
	fileStore := &vectorstore.FileStore{}
	embedder := fakeRuntimeEmbedder{}

	SetVectorStoreManager(manager)
	SetIngestionPipeline(pipeline)
	SetFileStore(fileStore)
	SetEmbedder(embedder)
	t.Cleanup(func() {
		SetVectorStoreManager(nil)
		SetIngestionPipeline(nil)
		SetFileStore(nil)
		SetEmbedder(nil)
	})

	apiServer := &ClassificationAPIServer{}
	if got := apiServer.currentVectorStoreManager(); got != manager {
		t.Fatalf("currentVectorStoreManager() = %v, want %v", got, manager)
	}
	if got := apiServer.currentVectorStorePipeline(); got != pipeline {
		t.Fatalf("currentVectorStorePipeline() = %v, want %v", got, pipeline)
	}
	if got := apiServer.currentFileStore(); got != fileStore {
		t.Fatalf("currentFileStore() = %v, want %v", got, fileStore)
	}
	if got := apiServer.currentVectorStoreEmbedder(); got != embedder {
		t.Fatalf("currentVectorStoreEmbedder() = %v, want %v", got, embedder)
	}
}

func TestRuntimeRegistrySuppressesLegacyVectorGlobalsUntilPublished(t *testing.T) {
	globalManager := &vectorstore.Manager{}
	globalPipeline := &vectorstore.IngestionPipeline{}
	globalFileStore := &vectorstore.FileStore{}
	globalEmbedder := fakeRuntimeEmbedder{}

	SetVectorStoreManager(globalManager)
	SetIngestionPipeline(globalPipeline)
	SetFileStore(globalFileStore)
	SetEmbedder(globalEmbedder)
	t.Cleanup(func() {
		SetVectorStoreManager(nil)
		SetIngestionPipeline(nil)
		SetFileStore(nil)
		SetEmbedder(nil)
	})

	registry := routerruntime.NewRegistry(nil)
	apiServer := &ClassificationAPIServer{runtimeRegistry: registry}

	if got := apiServer.currentVectorStoreManager(); got != nil {
		t.Fatalf("currentVectorStoreManager() = %v, want nil before runtime publication", got)
	}
	if got := apiServer.currentVectorStorePipeline(); got != nil {
		t.Fatalf("currentVectorStorePipeline() = %v, want nil before runtime publication", got)
	}
	if got := apiServer.currentFileStore(); got != nil {
		t.Fatalf("currentFileStore() = %v, want nil before runtime publication", got)
	}
	if got := apiServer.currentVectorStoreEmbedder(); got != nil {
		t.Fatalf("currentVectorStoreEmbedder() = %v, want nil before runtime publication", got)
	}

	runtimeManager := &vectorstore.Manager{}
	runtimePipeline := &vectorstore.IngestionPipeline{}
	runtimeFileStore := &vectorstore.FileStore{}
	runtimeEmbedder := fakeRuntimeEmbedder{}
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{
		Manager:   runtimeManager,
		Pipeline:  runtimePipeline,
		FileStore: runtimeFileStore,
		Embedder:  runtimeEmbedder,
	})

	if got := apiServer.currentVectorStoreManager(); got != runtimeManager {
		t.Fatalf("currentVectorStoreManager() = %v, want runtime manager %v", got, runtimeManager)
	}
	if got := apiServer.currentVectorStorePipeline(); got != runtimePipeline {
		t.Fatalf("currentVectorStorePipeline() = %v, want runtime pipeline %v", got, runtimePipeline)
	}
	if got := apiServer.currentFileStore(); got != runtimeFileStore {
		t.Fatalf("currentFileStore() = %v, want runtime file store %v", got, runtimeFileStore)
	}
	if got := apiServer.currentVectorStoreEmbedder(); got != runtimeEmbedder {
		t.Fatalf("currentVectorStoreEmbedder() = %v, want runtime embedder %v", got, runtimeEmbedder)
	}
}

func TestRuntimeRegistrySuppressesLegacySelectionGlobalUntilPublished(t *testing.T) {
	globalRegistry := selection.NewRegistry()
	originalRegistry := selection.GlobalRegistry
	selection.GlobalRegistry = globalRegistry
	t.Cleanup(func() {
		selection.GlobalRegistry = originalRegistry
	})

	registry := routerruntime.NewRegistry(nil)
	apiServer := &ClassificationAPIServer{runtimeRegistry: registry}

	if got := apiServer.currentSelectionRegistry(); got != nil {
		t.Fatalf("currentSelectionRegistry() = %v, want nil before runtime publication", got)
	}

	runtimeRegistry := selection.NewRegistry()
	registry.SetModelSelector(runtimeRegistry)
	if got := apiServer.currentSelectionRegistry(); got != runtimeRegistry {
		t.Fatalf("currentSelectionRegistry() = %v, want runtime registry %v", got, runtimeRegistry)
	}
}

type fakeRuntimeEmbedder struct{}

func (fakeRuntimeEmbedder) Embed(_ string) ([]float32, error) {
	return []float32{1, 0}, nil
}

func (fakeRuntimeEmbedder) Dimension() int {
	return 2
}
