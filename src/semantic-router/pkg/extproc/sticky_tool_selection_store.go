package extproc

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
)

func stickyToolSelectionEnabled(cfg *config.RouterConfig) bool {
	if cfg == nil {
		return false
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		plugin := decision.GetToolSelectionConfig()
		if plugin != nil && plugin.Sticky != nil && plugin.Sticky.Enabled {
			return true
		}
	}
	return false
}

func buildStickyToolSelectionManager(cfg *config.RouterConfig) (*sessiontools.Manager, sessiontools.Store, error) {
	if !stickyToolSelectionEnabled(cfg) {
		return nil, nil, nil
	}

	storeConfig := config.ToolSessionStoreConfig{}
	if cfg.ToolSessions != nil {
		storeConfig = *cfg.ToolSessions
	}
	if err := storeConfig.Validate(); err != nil {
		return nil, nil, err
	}
	var store sessiontools.Store
	switch storeConfig.EffectiveBackend() {
	case config.ToolSessionStoreBackendLocal:
		store = sessiontools.NewMemoryStore(storeConfig, nil)
	case config.ToolSessionStoreBackendRedis:
		redisStore, storeErr := sessiontools.NewRedisStore(storeConfig)
		if storeErr != nil {
			return nil, nil, storeErr
		}
		store = redisStore
	default:
		return nil, nil, fmt.Errorf(
			"tool_sessions store: backend %q is not supported by sticky tool selection runtime",
			storeConfig.EffectiveBackend(),
		)
	}
	options := sessiontools.DefaultManagerOptions()
	options.TTL = time.Duration(storeConfig.EffectiveTTLSeconds()) * time.Second
	options.MaxStateBytes = storeConfig.EffectiveMaxStateBytes()
	options.OperationTimeout = time.Duration(storeConfig.EffectiveTimeoutMs()) * time.Millisecond
	manager, err := sessiontools.NewManager(store, options)
	if err != nil {
		_ = store.Close()
		return nil, nil, err
	}
	return manager, store, nil
}
