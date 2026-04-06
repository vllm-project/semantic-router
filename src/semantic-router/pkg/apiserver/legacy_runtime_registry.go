//go:build !windows && cgo

package apiserver

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// newLegacyRuntimeRegistry bridges the remaining compatibility globals into a
// narrow runtime registry for legacy API-server entrypoints that do not receive
// a shared routerruntime.Registry directly.
func newLegacyRuntimeRegistry(configPath string) (*routerruntime.Registry, func(), error) {
	cfg, err := resolveLegacyRuntimeConfig(configPath)
	if err != nil {
		return nil, nil, err
	}

	registry := routerruntime.NewRegistry(cfg)
	syncLegacyRuntimeRegistry(registry, cfg)

	subscription := config.SubscribeConfigUpdates(1)
	go func() {
		for newCfg := range subscription.Updates() {
			if newCfg == nil {
				continue
			}
			syncLegacyRuntimeRegistry(registry, newCfg)
		}
	}()

	return registry, subscription.Close, nil
}

func resolveLegacyRuntimeConfig(configPath string) (*config.RouterConfig, error) {
	if cfg := config.Get(); cfg != nil {
		return cfg, nil
	}
	if configPath == "" {
		return nil, fmt.Errorf("configuration not initialized")
	}
	return config.Load(configPath)
}

func syncLegacyRuntimeRegistry(registry *routerruntime.Registry, cfg *config.RouterConfig) {
	if registry == nil || cfg == nil {
		return
	}

	registry.SetClassificationService(services.GetGlobalClassificationService())
	registry.SetMemoryStore(memory.GetGlobalMemoryStore())
	registry.SetVectorStoreRuntime(snapshotLegacyVectorStoreRuntime())
	registry.RefreshRuntimeConfig(cfg)
}

func snapshotLegacyVectorStoreRuntime() *routerruntime.VectorStoreRuntime {
	if vectorStoreManager == nil && globalPipeline == nil && globalEmbedder == nil && globalFileStore == nil {
		return nil
	}

	return &routerruntime.VectorStoreRuntime{
		Manager:   vectorStoreManager,
		Pipeline:  globalPipeline,
		Embedder:  globalEmbedder,
		FileStore: globalFileStore,
	}
}
