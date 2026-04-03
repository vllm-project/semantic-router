package routerruntime

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// Registry is the narrow runtime-owned dependency seam shared by startup,
// reload, extproc, and the API server.
type Registry struct {
	mu                    sync.RWMutex
	config                *config.RouterConfig
	classificationService *services.ClassificationService
	memoryStore           memory.Store
	vectorStore           *VectorStoreRuntime
}

func NewRegistry(cfg *config.RouterConfig) *Registry {
	return &Registry{config: cfg}
}

func (r *Registry) CurrentConfig() *config.RouterConfig {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.config
}

func (r *Registry) UpdateConfig(cfg *config.RouterConfig) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.config = cfg
	r.mu.Unlock()
}

func (r *Registry) ClassificationService() *services.ClassificationService {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.classificationService
}

func (r *Registry) SetClassificationService(service *services.ClassificationService) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.classificationService = service
	r.mu.Unlock()
}

func (r *Registry) MemoryStore() memory.Store {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.memoryStore
}

func (r *Registry) SetMemoryStore(store memory.Store) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.memoryStore = store
	r.mu.Unlock()
}

func (r *Registry) VectorStoreRuntime() *VectorStoreRuntime {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.vectorStore
}

func (r *Registry) SetVectorStoreRuntime(runtime *VectorStoreRuntime) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.vectorStore = runtime
	r.mu.Unlock()
}

func (r *Registry) PublishRouterRuntime(
	cfg *config.RouterConfig,
	classificationService *services.ClassificationService,
	memoryStore memory.Store,
) {
	if r == nil {
		return
	}
	r.mu.Lock()
	if cfg != nil {
		r.config = cfg
	}
	r.classificationService = classificationService
	r.memoryStore = memoryStore
	r.mu.Unlock()
}

func (r *Registry) RefreshRuntimeConfig(newCfg *config.RouterConfig) {
	if r == nil {
		return
	}
	r.UpdateConfig(newCfg)
	if service := r.ClassificationService(); service != nil {
		service.RefreshRuntimeConfig(newCfg)
	}
}
