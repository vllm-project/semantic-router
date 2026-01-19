package responsestore

import (
	"fmt"
)

// NewStore creates a new store based on the configuration.
func NewStore(config StoreConfig) (CombinedStore, error) {
	if !config.Enabled {
		// Return a disabled memory store
		return NewMemoryStore(StoreConfig{Enabled: false})
	}

	switch config.BackendType {
	case MemoryStoreType, "":
		return NewMemoryStore(config)
	case MilvusStoreType:
		return NewMilvusStore(config)
	case RedisStoreType:
		return NewRedisStore(config)
	default:
		return nil, fmt.Errorf("unknown store backend type: %s", config.BackendType)
	}
}

// NewMilvusStore creates a new Milvus-based store.
// This is a placeholder that will be implemented in a separate file.
func NewMilvusStore(config StoreConfig) (CombinedStore, error) {
	return nil, fmt.Errorf("milvus store not yet implemented")
}
