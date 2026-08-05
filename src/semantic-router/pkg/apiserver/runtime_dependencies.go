//go:build !windows && cgo

package apiserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func (s *ClassificationAPIServer) currentMemoryStore() memory.Store {
	if s != nil && s.runtimeRegistry != nil {
		if store := s.runtimeRegistry.MemoryStore(); store != nil {
			return store
		}
	}
	if s == nil {
		return nil
	}
	return s.memoryStore
}

func (s *ClassificationAPIServer) currentVectorStoreRuntime() *routerruntime.VectorStoreRuntime {
	if s == nil || s.runtimeRegistry == nil {
		return nil
	}
	return s.runtimeRegistry.VectorStoreRuntime()
}

func (s *ClassificationAPIServer) currentVectorStoreManager() *vectorstore.Manager {
	if runtime := s.currentVectorStoreRuntime(); runtime != nil {
		return runtime.Manager
	}
	if s != nil && s.runtimeRegistry != nil {
		return nil
	}
	return GetVectorStoreManager()
}

func (s *ClassificationAPIServer) currentVectorStoreEmbedder() vectorstore.Embedder {
	if runtime := s.currentVectorStoreRuntime(); runtime != nil {
		return runtime.Embedder
	}
	if s != nil && s.runtimeRegistry != nil {
		return nil
	}
	return GetEmbedder()
}

func (s *ClassificationAPIServer) currentVectorStorePipeline() *vectorstore.IngestionPipeline {
	if runtime := s.currentVectorStoreRuntime(); runtime != nil {
		return runtime.Pipeline
	}
	if s != nil && s.runtimeRegistry != nil {
		return nil
	}
	return globalRuntimeDeps.getIngestionPipeline()
}

func (s *ClassificationAPIServer) currentFileStore() *vectorstore.FileStore {
	if runtime := s.currentVectorStoreRuntime(); runtime != nil {
		return runtime.FileStore
	}
	if s != nil && s.runtimeRegistry != nil {
		return nil
	}
	return globalRuntimeDeps.getFileStore()
}

func (s *ClassificationAPIServer) currentSelectionRegistry() *selection.Registry {
	if s != nil && s.runtimeRegistry != nil {
		if registry := s.runtimeRegistry.ModelSelector(); registry != nil {
			return registry
		}
		return nil
	}
	return selection.GlobalRegistry
}

func (s *ClassificationAPIServer) currentLearningRuntime() routerruntime.LearningRuntime {
	if s != nil && s.runtimeRegistry != nil {
		return s.runtimeRegistry.LearningRuntime()
	}
	return nil
}
