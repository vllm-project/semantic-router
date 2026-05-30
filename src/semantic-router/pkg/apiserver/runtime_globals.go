//go:build !windows && cgo

package apiserver

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

var globalRuntimeDeps runtimeDependencyGlobals

type runtimeDependencyGlobals struct {
	mu sync.RWMutex

	vectorStoreManager *vectorstore.Manager
	pipeline           *vectorstore.IngestionPipeline
	embedder           vectorstore.Embedder
	fileStore          *vectorstore.FileStore
}

func (g *runtimeDependencyGlobals) setVectorStoreManager(manager *vectorstore.Manager) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.vectorStoreManager = manager
}

func (g *runtimeDependencyGlobals) getVectorStoreManager() *vectorstore.Manager {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.vectorStoreManager
}

func (g *runtimeDependencyGlobals) setIngestionPipeline(pipeline *vectorstore.IngestionPipeline) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.pipeline = pipeline
}

func (g *runtimeDependencyGlobals) getIngestionPipeline() *vectorstore.IngestionPipeline {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.pipeline
}

func (g *runtimeDependencyGlobals) setEmbedder(embedder vectorstore.Embedder) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.embedder = embedder
}

func (g *runtimeDependencyGlobals) getEmbedder() vectorstore.Embedder {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.embedder
}

func (g *runtimeDependencyGlobals) setFileStore(fileStore *vectorstore.FileStore) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.fileStore = fileStore
}

func (g *runtimeDependencyGlobals) getFileStore() *vectorstore.FileStore {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.fileStore
}
