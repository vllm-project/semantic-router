package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

// createResponseStore creates a response store based on configuration.
func createResponseStore(cfg *config.RouterConfig) (responsestore.ResponseStore, error) {
	storeConfig := responsestore.StoreConfig{
		Enabled:     true,
		TTLSeconds:  cfg.ResponseAPI.TTLSeconds,
		BackendType: responsestore.StoreBackendType(cfg.ResponseAPI.StoreBackend),
		Memory: responsestore.MemoryStoreConfig{
			MaxResponses: cfg.ResponseAPI.MaxResponses,
		},
		Milvus: responsestore.MilvusStoreConfig{
			Address:            cfg.ResponseAPI.Milvus.Address,
			Database:           cfg.ResponseAPI.Milvus.Database,
			ResponseCollection: cfg.ResponseAPI.Milvus.Collection,
		},
		Redis: responsestore.RedisStoreConfig{
			Address:          cfg.ResponseAPI.Redis.Address,
			Password:         cfg.ResponseAPI.Redis.Password,
			DB:               cfg.ResponseAPI.Redis.DB,
			KeyPrefix:        cfg.ResponseAPI.Redis.KeyPrefix,
			ClusterMode:      cfg.ResponseAPI.Redis.ClusterMode,
			ClusterAddresses: cfg.ResponseAPI.Redis.ClusterAddresses,
			PoolSize:         cfg.ResponseAPI.Redis.PoolSize,
			MinIdleConns:     cfg.ResponseAPI.Redis.MinIdleConns,
			MaxRetries:       cfg.ResponseAPI.Redis.MaxRetries,
			DialTimeout:      cfg.ResponseAPI.Redis.DialTimeout,
			ReadTimeout:      cfg.ResponseAPI.Redis.ReadTimeout,
			WriteTimeout:     cfg.ResponseAPI.Redis.WriteTimeout,
			TLSEnabled:       cfg.ResponseAPI.Redis.TLSEnabled,
			TLSCertPath:      cfg.ResponseAPI.Redis.TLSCertPath,
			TLSKeyPath:       cfg.ResponseAPI.Redis.TLSKeyPath,
			TLSCAPath:        cfg.ResponseAPI.Redis.TLSCAPath,
			ConfigPath:       cfg.ResponseAPI.Redis.ConfigPath,
		},
	}

	return responsestore.NewStore(storeConfig)
}
