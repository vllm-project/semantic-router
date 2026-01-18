package store

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// NewStore creates a new replay store based on the configuration.
// If the store is disabled, returns a disabled memory store.
func NewStore(config StoreConfig) (ReplayStore, error) {
	if !config.Enabled {
		logging.Debugf("Replay store disabled, using disabled memory store")
		return NewMemoryStore(MemoryStoreConfig{}, 0, false), nil
	}

	// Apply TTL defaults
	ttlSeconds := config.TTLSeconds
	if ttlSeconds <= 0 {
		ttlSeconds = int(DefaultTTL.Seconds())
	}

	switch config.BackendType {
	case MemoryStoreType, "":
		logging.Infof("Creating memory replay store with max_records=%d, ttl=%ds",
			config.Memory.MaxRecords, ttlSeconds)

		memConfig := config.Memory
		if memConfig.MaxRecords <= 0 {
			memConfig.MaxRecords = config.MaxRecords
		}
		return NewMemoryStore(memConfig, ttlSeconds, true), nil

	case RedisStoreType:
		logging.Infof("Creating Redis replay store at %s", config.Redis.Address)

		redisConfig := config.Redis
		if redisConfig.TTLSeconds <= 0 {
			redisConfig.TTLSeconds = ttlSeconds
		}
		return NewRedisStore(redisConfig)

	default:
		return nil, fmt.Errorf("unknown store backend type: %s (supported: memory, redis)", config.BackendType)
	}
}

// ValidateConfig validates the store configuration.
func ValidateConfig(config StoreConfig) error {
	if !config.Enabled {
		return nil
	}

	switch config.BackendType {
	case MemoryStoreType, "":
		// Memory store has no required configuration
		return nil

	case RedisStoreType:
		if config.Redis.Address == "" {
			return fmt.Errorf("redis address is required for redis backend")
		}
		return nil

	default:
		return fmt.Errorf("unknown store backend type: %s", config.BackendType)
	}
}
