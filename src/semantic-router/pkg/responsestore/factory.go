package responsestore

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NewStore creates a new store based on the configuration.
func NewStore(config StoreConfig) (CombinedStore, error) {
	if !config.Enabled {
		return NewMemoryStore(StoreConfig{Enabled: false})
	}

	switch config.BackendType {
	case MemoryStoreType, "":
		logging.Warnf("Response API store_backend is set to %q — all response and conversation "+
			"history will be lost on router restart. Use \"redis\" for durable storage in production.",
			"memory")
		return NewMemoryStore(config)
	case RedisStoreType:
		return NewRedisStore(config)
	default:
		return nil, fmt.Errorf("unknown store backend type: %s", config.BackendType)
	}
}
