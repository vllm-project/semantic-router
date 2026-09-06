package routerruntime

import (
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
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
	modelSelector         *selection.Registry
	learningRuntime       LearningRuntime
	replayRuntime         ReplayRuntime
	responseCache         *cache.ResponseCacheService
	contextCompression    *contextcompression.Service
	compressionRecovery   contextcompression.RecoveryStore
}

// RouterRuntimeSnapshot is the router-owned management surface published as
// one generation. Keeping these dependencies in one atomic snapshot prevents
// API handlers from observing a new config with an old cache or learning
// runtime during hot reload.
type RouterRuntimeSnapshot struct {
	Config                *config.RouterConfig
	ClassificationService *services.ClassificationService
	MemoryStore           memory.Store
	ModelSelector         *selection.Registry
	LearningRuntime       LearningRuntime
	ReplayRuntime         ReplayRuntime
	ResponseCache         *cache.ResponseCacheService
	ContextCompression    *contextcompression.Service
	CompressionRecovery   contextcompression.RecoveryStore
}

func (r *Registry) ContextCompression() (
	*contextcompression.Service,
	contextcompression.RecoveryStore,
) {
	if r == nil {
		return nil, nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.contextCompression, r.compressionRecovery
}

func (r *Registry) AcquireContextCompression() (
	*contextcompression.Service,
	contextcompression.RecoveryStore,
	func(),
) {
	if r == nil {
		return nil, nil, func() {}
	}
	r.mu.RLock()
	release, acquired := acquireRuntimeLease(r.learningRuntime)
	if !acquired && r.learningRuntime != nil {
		r.mu.RUnlock()
		return nil, nil, func() {}
	}
	service, recovery := r.contextCompression, r.compressionRecovery
	r.mu.RUnlock()
	return service, recovery, release
}

func (r *Registry) SetContextCompression(
	service *contextcompression.Service,
	recovery contextcompression.RecoveryStore,
) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.contextCompression = service
	r.compressionRecovery = recovery
	r.mu.Unlock()
}

func (r *Registry) ResponseCache() *cache.ResponseCacheService {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.responseCache
}

func (r *Registry) AcquireResponseCache() (*cache.ResponseCacheService, func()) {
	if r == nil {
		return nil, func() {}
	}
	r.mu.RLock()
	release, acquired := acquireRuntimeLease(r.learningRuntime)
	if !acquired && r.learningRuntime != nil {
		r.mu.RUnlock()
		return nil, func() {}
	}
	service := r.responseCache
	r.mu.RUnlock()
	return service, release
}

func (r *Registry) SetResponseCache(service *cache.ResponseCacheService) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.responseCache = service
	r.mu.Unlock()
}

// LearningRuntime is the narrow API-server seam for Router Learning state.
// The implementation lives with the router runtime; the API server only needs
// to forward typed outcomes without depending on extproc internals.
type LearningRuntime interface {
	OutcomeRuntime
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

func (r *Registry) ModelSelector() *selection.Registry {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.modelSelector
}

func (r *Registry) SetModelSelector(registry *selection.Registry) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.modelSelector = registry
	r.mu.Unlock()
}

func (r *Registry) LearningRuntime() LearningRuntime {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.learningRuntime
}

// AcquireReplayRuntime returns the currently published replay runtime and a
// release function that keeps its router generation alive for the duration of
// one authenticated management request.
func (r *Registry) AcquireReplayRuntime() (ReplayRuntime, func()) {
	if r == nil {
		return nil, func() {}
	}
	r.mu.RLock()
	runtime := r.replayRuntime
	if runtime == nil {
		r.mu.RUnlock()
		return nil, func() {}
	}
	release, acquired := acquireRuntimeLease(runtime)
	r.mu.RUnlock()
	if !acquired {
		return nil, func() {}
	}
	return runtime, release
}

func (r *Registry) SetReplayRuntime(runtime ReplayRuntime) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.replayRuntime = runtime
	r.mu.Unlock()
}

// AcquireLearningRuntime returns the currently published learning runtime and
// a release function that keeps its router generation alive for the duration
// of a management request. Legacy test/runtime implementations that do not
// own closeable resources retain the previous no-op lease behavior.
func (r *Registry) AcquireLearningRuntime() (LearningRuntime, func()) {
	if r == nil {
		return nil, func() {}
	}
	r.mu.RLock()
	runtime := r.learningRuntime
	if runtime == nil {
		r.mu.RUnlock()
		return nil, func() {}
	}
	release, acquired := acquireRuntimeLease(runtime)
	r.mu.RUnlock()
	if !acquired {
		return nil, func() {}
	}
	return runtime, release
}

func acquireRuntimeLease(runtime interface{}) (func(), bool) {
	if runtime == nil {
		return func() {}, true
	}
	if leased, ok := runtime.(interface {
		AcquireLease() (func(), bool)
	}); ok {
		return leased.AcquireLease()
	}
	return func() {}, true
}

func (r *Registry) SetLearningRuntime(runtime LearningRuntime) {
	if r == nil {
		return
	}
	r.mu.Lock()
	r.learningRuntime = runtime
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

func (r *Registry) PublishRouterRuntimeSnapshot(snapshot RouterRuntimeSnapshot) {
	if r == nil {
		return
	}
	r.mu.Lock()
	if snapshot.Config != nil {
		r.config = snapshot.Config
	}
	r.classificationService = snapshot.ClassificationService
	r.memoryStore = snapshot.MemoryStore
	r.modelSelector = snapshot.ModelSelector
	r.learningRuntime = snapshot.LearningRuntime
	r.replayRuntime = snapshot.ReplayRuntime
	r.responseCache = snapshot.ResponseCache
	r.contextCompression = snapshot.ContextCompression
	r.compressionRecovery = snapshot.CompressionRecovery
	r.mu.Unlock()
}

func (r *Registry) RefreshRuntimeConfig(newCfg *config.RouterConfig) {
	if r == nil {
		return
	}
	if service := r.ClassificationService(); service != nil {
		if err := service.TryRefreshRuntimeConfig(newCfg); err != nil {
			logging.Errorf(
				"Runtime config refresh rejected; retaining previous registry snapshot: %v",
				err,
			)
			return
		}
	}
	r.UpdateConfig(newCfg)
}
