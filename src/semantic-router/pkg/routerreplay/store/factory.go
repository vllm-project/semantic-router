package store

import (
	"fmt"
)

// NewStorage creates a new storage backend based on the provided configuration.
func NewStorage(cfg *Config) (Storage, error) {
	if cfg == nil {
		return nil, fmt.Errorf("storage config is required")
	}

	backend := cfg.Backend
	if backend == "" {
		backend = "memory"
	}

	return newStorageBackend(backend, cfg)
}

func newStorageBackend(backend string, cfg *Config) (Storage, error) {
	switch backend {
	case "memory":
		return NewMemoryStore(200, cfg.TTLSeconds), nil

	case "redis":
		if cfg.Redis == nil {
			return nil, fmt.Errorf("redis config required when backend is 'redis'")
		}
		return NewRedisStore(cfg.Redis, cfg.TTLSeconds, cfg.AsyncWrites)

	case "postgres":
		if cfg.Postgres == nil {
			return nil, fmt.Errorf("postgres config required when backend is 'postgres'")
		}
		return NewPostgresStore(cfg.Postgres, cfg.TTLSeconds, cfg.AsyncWrites)

	case "milvus":
		if cfg.Milvus == nil {
			return nil, fmt.Errorf("milvus config required when backend is 'milvus'")
		}
		return NewMilvusStore(cfg.Milvus, cfg.TTLSeconds, cfg.AsyncWrites)

	case "qdrant":
		if cfg.Qdrant == nil {
			return nil, fmt.Errorf("qdrant config required when backend is 'qdrant'")
		}
		return NewQdrantStore(cfg.Qdrant, cfg.TTLSeconds, cfg.AsyncWrites)

	default:
		return nil, fmt.Errorf("unknown storage backend: %s (supported: memory, redis, postgres, milvus, qdrant)", backend)
	}
}
