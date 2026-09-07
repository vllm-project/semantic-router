package extproc

import (
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (components *routerComponents) buildMemoryRuntime(createStore func(*config.RouterConfig) (memory.Store, error)) error {
	var err error
	components.memoryStore, components.memoryExtractor, err = createMemoryRuntime(components.cfg, createStore)
	if err != nil {
		return rollbackResources(components.resources, err)
	}
	if components.memoryStore != nil {
		components.resources.add(components.memoryStore.Close)
	}
	return nil
}

func createMemoryRuntime(
	cfg *config.RouterConfig,
	createStore func(*config.RouterConfig) (memory.Store, error),
) (memory.Store, *memory.MemoryExtractor, error) {
	if !isMemoryEnabled(cfg) {
		return nil, nil, nil
	}

	// publishRouterState publishes the store after the candidate commits.
	memoryStore, err := createStore(cfg)
	if err != nil {
		var mismatch *milvuslifecycle.VectorDimensionMismatchError
		if errors.As(err, &mismatch) {
			return nil, nil, fmt.Errorf("failed to initialize Memory: %w", err)
		}
		logging.Warnf("Failed to create memory store: %v, Memory will be disabled", err)
		return nil, nil, nil
	}

	backend := cfg.Memory.Backend
	if backend == "" {
		backend = "milvus"
	}
	if rc := cfg.Memory.RedisCache; rc != nil && rc.Enabled && rc.Address != "" {
		logging.Infof("Memory enabled with %s backend and Redis hot cache", backend)
	} else {
		logging.Infof("Memory enabled with %s backend", backend)
	}

	memoryExtractor := memory.NewMemoryChunkStore(memoryStore)
	if memoryExtractor != nil {
		logging.Infof("Memory chunk store enabled (direct conversation storage)")
	}

	return memoryStore, memoryExtractor, nil
}
