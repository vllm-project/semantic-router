package cache

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

const (
	mmbertMemoryCacheLayer     = 6
	mmbertMemoryCacheDimension = 256
	// Cache identities use the extracted semantic query, with model-window
	// rejection before inference. Changing this text policy requires a new version.
	semanticCacheInputPolicy = "response-cache-semantic-query:model-window-reject:v1"
)

// LocalEmbeddingSettings mirrors the backend's actual embedding call. Unsupported
// providers do not acquire a guessed content identity during this migration.
func LocalEmbeddingSettings(backend CacheBackend) (embedding.ConsumerSettings, bool, error) {
	var model string
	var layer, dimension int
	switch value := backend.(type) {
	case *InMemoryCache:
		model, layer, dimension = value.embeddingModel, mmbertMemoryCacheLayer, mmbertMemoryCacheDimension
	case *RedisCache:
		model = value.embeddingModel
		var err error
		dimension, err = value.embeddingDimension()
		if err != nil {
			return embedding.ConsumerSettings{}, false, fmt.Errorf("resolve redis cache embedding dimension: %w", err)
		}
	case *ValkeyCache:
		model = value.embeddingModel
		var err error
		dimension, err = value.embeddingDimension()
		if err != nil {
			return embedding.ConsumerSettings{}, false, fmt.Errorf("resolve valkey cache embedding dimension: %w", err)
		}
	case *MilvusCache:
		model = value.embeddingModel
		var err error
		dimension, err = value.embeddingDimension()
		if err != nil {
			return embedding.ConsumerSettings{}, false, fmt.Errorf("resolve milvus cache embedding dimension: %w", err)
		}
	case *QdrantCache:
		model = value.embeddingModel
		var err error
		dimension, err = value.embeddingDimension()
		if err != nil {
			return embedding.ConsumerSettings{}, false, fmt.Errorf("resolve qdrant cache embedding dimension: %w", err)
		}
	case *HybridCache:
		if value.milvusCache == nil {
			return embedding.ConsumerSettings{}, false, nil
		}
		return LocalEmbeddingSettings(value.milvusCache)
	default:
		return embedding.ConsumerSettings{}, false, nil
	}
	if model != "mmbert" {
		return embedding.ConsumerSettings{}, false, nil
	}
	return embedding.ConsumerSettings{ModelType: model, Layer: layer, Dimension: dimension, InputPolicy: semanticCacheInputPolicy}, true, nil
}

// PrepareEmbeddingNamespace runs before any persistent index is opened. The
// returned copy is bound to the loaded representation; the user's logical
// configuration and all prior namespaces remain untouched.
func PrepareEmbeddingNamespace(cfg CacheConfig, resolve func(embedding.ConsumerSettings) (embedding.ContentIdentity, error)) (CacheConfig, string, error) {
	if !cfg.Enabled || normalizeEmbeddingModel(cfg.EmbeddingModel) != "mmbert" {
		return cfg, "", nil
	}
	if err := ValidateCacheConfig(cfg); err != nil {
		return cfg, "", err
	}
	backend := normalizedBackendType(cfg.BackendType)
	// These views reuse exactly the same per-backend dimension calculation as
	// inference, without connecting to the old physical index first.
	var view CacheBackend
	var logical []string
	switch backend {
	case InMemoryCacheType:
		view = &InMemoryCache{embeddingModel: "mmbert"}
	case RedisCacheType:
		view = &RedisCache{embeddingModel: "mmbert", config: cfg.Redis, embeddingProvider: cfg.EmbeddingProvider}
		logical = []string{cfg.Redis.Index.Name, cfg.Redis.Index.Prefix}
	case ValkeyCacheType:
		view = &ValkeyCache{embeddingModel: "mmbert", config: cfg.Valkey, embeddingProvider: cfg.EmbeddingProvider}
		logical = []string{cfg.Valkey.Index.Name, cfg.Valkey.Index.Prefix}
	case MilvusCacheType, HybridCacheType:
		view = &MilvusCache{
			embeddingModel:    "mmbert",
			config:            cfg.Milvus,
			embeddingProvider: cfg.EmbeddingProvider,
		}
		logical = []string{cfg.Milvus.Collection.Name}
	case QdrantCacheType:
		view = &QdrantCache{embeddingModel: "mmbert", embeddingProvider: cfg.EmbeddingProvider}
		name := cfg.Qdrant.CollectionName
		if name == "" {
			name = "semantic_cache"
		}
		logical = []string{name}
	}
	settings, supported, err := LocalEmbeddingSettings(view)
	if err != nil {
		return cfg, "", fmt.Errorf("resolve cache embedding settings: %w", err)
	}
	if !supported {
		return cfg, "", fmt.Errorf("unsupported local embedding cache backend %s", backend)
	}
	identity, err := resolve(settings)
	if err != nil {
		return cfg, "", err
	}
	if identity.Fingerprint == "" {
		return cfg, "", fmt.Errorf("embedding content identity is empty")
	}
	logical = append([]string{string(backend), identity.Fingerprint}, logical...)
	encoded, err := json.Marshal(logical)
	if err != nil {
		return cfg, "", err
	}
	scope := fmt.Sprintf("%x", sha256.Sum256(encoded))
	name := "semantic_cache_" + scope
	prefix := "vsr:embedding-cache:" + scope + ":"
	switch backend {
	case RedisCacheType:
		copy := *cfg.Redis
		cfg.Redis = &copy
		copy.Index.Name, copy.Index.Prefix = name, prefix
	case ValkeyCacheType:
		copy := *cfg.Valkey
		cfg.Valkey = &copy
		copy.Index.Name, copy.Index.Prefix = name, prefix
	case MilvusCacheType, HybridCacheType:
		copy := *cfg.Milvus
		cfg.Milvus = &copy
		copy.Collection.Name = name
	case QdrantCacheType:
		copy := *cfg.Qdrant
		cfg.Qdrant = &copy
		copy.CollectionName = name
	}
	return cfg, identity.Fingerprint, nil
}
