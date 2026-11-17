package cache

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// Cache identities use the extracted semantic query, with model-window
// rejection before inference. Changing this text policy requires a new version.
const semanticCacheInputPolicy = "response-cache-semantic-query:model-window-reject:v1"

// LocalEmbeddingSettings mirrors the backend's actual embedding call: its
// deployment and the view it reads.
func LocalEmbeddingSettings(backend CacheBackend) (embedding.ConsumerSettings, bool) {
	var deployment string
	var layer, dimension int
	switch value := backend.(type) {
	case *InMemoryCache:
		deployment, layer = value.embeddingDeployment, value.embeddingView.Layer
		dimension = semanticCacheEmbeddingDimension(value.embeddingView.Dimension, value.embeddingProvider)
	case *RedisCache:
		deployment, dimension = value.embeddingDeployment, value.embeddingDimension()
	case *ValkeyCache:
		deployment, dimension = value.embeddingDeployment, value.embeddingDimension()
	case *MilvusCache:
		deployment, dimension = value.embeddingDeployment, value.embeddingDimension()
	case *QdrantCache:
		deployment, dimension = value.embeddingDeployment, value.embeddingDimension()
	case *HybridCache:
		if value.milvusCache == nil {
			return embedding.ConsumerSettings{}, false
		}
		return LocalEmbeddingSettings(value.milvusCache)
	default:
		return embedding.ConsumerSettings{}, false
	}
	if deployment == "" {
		return embedding.ConsumerSettings{}, false
	}
	return embedding.ConsumerSettings{ModelType: deployment, Layer: layer, Dimension: dimension, InputPolicy: semanticCacheInputPolicy}, true
}

// PrepareEmbeddingNamespace runs before any persistent index is opened. The
// returned copy is bound to the loaded representation; the user's logical
// configuration and all prior namespaces remain untouched. A provider without
// a content identity (a remote endpoint) keeps the configured namespace.
func PrepareEmbeddingNamespace(cfg CacheConfig, resolve func(embedding.ConsumerSettings) (embedding.ContentIdentity, error)) (CacheConfig, string, error) {
	if !cfg.Enabled || cfg.EmbeddingDeployment == "" {
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
		view = &InMemoryCache{embeddingDeployment: cfg.EmbeddingDeployment, embeddingView: embedding.Options{Dimension: cfg.EmbeddingDimension, Layer: cfg.EmbeddingLayer}, embeddingProvider: cfg.EmbeddingProvider}
	case RedisCacheType:
		view = &RedisCache{embeddingDeployment: cfg.EmbeddingDeployment, embeddingProvider: cfg.EmbeddingProvider, config: cfg.Redis}
		logical = []string{cfg.Redis.Index.Name, cfg.Redis.Index.Prefix}
	case ValkeyCacheType:
		view = &ValkeyCache{embeddingDeployment: cfg.EmbeddingDeployment, embeddingProvider: cfg.EmbeddingProvider, config: cfg.Valkey}
		logical = []string{cfg.Valkey.Index.Name, cfg.Valkey.Index.Prefix}
	case MilvusCacheType, HybridCacheType:
		view = &MilvusCache{embeddingDeployment: cfg.EmbeddingDeployment, embeddingProvider: cfg.EmbeddingProvider, config: cfg.Milvus}
		logical = []string{cfg.Milvus.Collection.Name}
	case QdrantCacheType:
		view = &QdrantCache{embeddingDeployment: cfg.EmbeddingDeployment, embeddingProvider: cfg.EmbeddingProvider}
		name := cfg.Qdrant.CollectionName
		if name == "" {
			name = "semantic_cache"
		}
		logical = []string{name}
	}
	settings, supported := LocalEmbeddingSettings(view)
	if !supported {
		return cfg, "", fmt.Errorf("unsupported local embedding cache backend %s", backend)
	}
	identity, err := resolve(settings)
	if errors.Is(err, embedding.ErrIdentityUnsupported) {
		return cfg, "", nil
	}
	if err != nil {
		return cfg, "", err
	}
	if identity.Fingerprint == "" || identity.Descriptor.Dimension <= 0 {
		return cfg, "", fmt.Errorf("embedding content identity lacks fingerprint or output dimension")
	}
	if settings.Dimension > 0 && settings.Dimension != identity.Descriptor.Dimension {
		return cfg, "", fmt.Errorf("cache dimension %d differs from prepared representation %d", settings.Dimension, identity.Descriptor.Dimension)
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
		copy.Index.VectorField.Dimension = identity.Descriptor.Dimension
	case ValkeyCacheType:
		copy := *cfg.Valkey
		cfg.Valkey = &copy
		copy.Index.Name, copy.Index.Prefix = name, prefix
		copy.Index.VectorField.Dimension = identity.Descriptor.Dimension
	case MilvusCacheType, HybridCacheType:
		copy := *cfg.Milvus
		cfg.Milvus = &copy
		copy.Collection.Name = name
		copy.Collection.VectorField.Dimension = identity.Descriptor.Dimension
	case QdrantCacheType:
		copy := *cfg.Qdrant
		cfg.Qdrant = &copy
		copy.CollectionName = name
	}
	return cfg, identity.Fingerprint, nil
}
