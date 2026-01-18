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

	switch backend {
	case "memory":
		maxRecords := 200
		if cfg.MaxBodyBytes > 0 {
			maxRecords = cfg.MaxBodyBytes
		}
		return NewMemoryStore(maxRecords, cfg.TTLSeconds), nil

	case "redis":
		if cfg.Redis == nil {
			return nil, fmt.Errorf("redis config required when backend is 'redis'")
		}
		store, err := NewRedisStore(cfg.Redis, cfg.TTLSeconds, cfg.AsyncWrites)
		if err != nil {
			return nil, err
		}
		return store, nil

	case "postgres":
		if cfg.Postgres == nil {
			return nil, fmt.Errorf("postgres config required when backend is 'postgres'")
		}
		store, err := NewPostgresStore(cfg.Postgres, cfg.TTLSeconds, cfg.AsyncWrites)
		if err != nil {
			return nil, err
		}
		return store, nil

	case "milvus":
		if cfg.Milvus == nil {
			return nil, fmt.Errorf("milvus config required when backend is 'milvus'")
		}
		store, err := NewMilvusStore(cfg.Milvus, cfg.TTLSeconds, cfg.AsyncWrites)
		if err != nil {
			return nil, err
		}
		return store, nil

	default:
		return nil, fmt.Errorf("unknown storage backend: %s (supported: memory, redis, postgres, milvus)", backend)
	}
}
