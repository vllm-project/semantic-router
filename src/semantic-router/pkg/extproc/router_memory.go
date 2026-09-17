package extproc

import (
	"context"
	"fmt"
	"time"

	"github.com/qdrant/go-client/qdrant"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func createMemoryRuntime(cfg *config.RouterConfig, sets ...*embedding.Set) (memory.Store, *memory.MemoryExtractor) {
	if !isMemoryEnabled(cfg) {
		return nil, nil
	}

	// publishRouterState publishes the store after the candidate commits.
	memoryStore, err := createMemoryStore(cfg, sets...)
	if err != nil {
		logging.Warnf("Failed to create memory store: %v, Memory will be disabled", err)
		return nil, nil
	}

	backend := cfg.Memory.Backend
	if backend == "" {
		backend = "milvus"
	}
	if rc := cfg.Memory.RedisCache; rc != nil && rc.Enabled && rc.Address != "" {
		logging.Infof("Memory enabled with %s backend and Redis hot cache", backend)
	} else {
		logging.Infof("Memory enabled with %s backend", backend)
	}

	memoryExtractor := memory.NewMemoryChunkStore(memoryStore)
	if memoryExtractor != nil {
		logging.Infof("Memory chunk store enabled (direct conversation storage)")
	}

	return memoryStore, memoryExtractor
}

func isMemoryEnabled(cfg *config.RouterConfig) bool {
	if cfg.Memory.Enabled {
		return true
	}

	for _, decision := range cfg.AllRoutingDecisions() {
		if memoryConfig := decision.GetMemoryConfig(); memoryConfig != nil && memoryConfig.Enabled {
			logging.Infof("Memory auto-enabled: decision '%s' uses memory plugin", decision.Name)
			return true
		}
	}

	return false
}

// createMemoryStore creates a memory store based on configuration.
// Switches on cfg.Memory.Backend: "valkey" creates a ValkeyStore, "milvus" (or empty) creates a MilvusStore.
func createMemoryStore(cfg *config.RouterConfig, sets ...*embedding.Set) (memory.Store, error) {
	bound, bindErr := bindMemoryEmbedding(cfg, sets...)
	if bindErr != nil {
		return nil, bindErr
	}
	cfg = bound
	backend := cfg.Memory.Backend
	if backend == "" {
		backend = "milvus"
	}

	var store memory.Store
	var err error

	switch backend {
	case "valkey":
		store, err = createValkeyMemoryStore(cfg, sets...)
	case "milvus":
		store, err = createMilvusMemoryStore(cfg, sets...)
	case "qdrant":
		store, err = createQdrantMemoryStore(cfg, sets...)
	default:
		return nil, fmt.Errorf("unsupported memory backend: %q (supported: milvus, valkey, qdrant)", backend)
	}

	if err != nil {
		return nil, err
	}

	// Optionally wrap with Redis hot cache
	result := wrapWithRedisCache(store, cfg, backend)
	return result, nil
}

// wrapWithRedisCache optionally wraps a memory store with a Redis hot cache.
// Returns the original store if caching is not configured or connection fails.
func wrapWithRedisCache(store memory.Store, cfg *config.RouterConfig, backend string) memory.Store {
	rc := cfg.Memory.RedisCache
	if rc == nil || !rc.Enabled || rc.Address == "" {
		return store
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cacheCfg := &memory.RedisCacheConfig{
		Address:    rc.Address,
		Password:   rc.Password,
		DB:         rc.DB,
		KeyPrefix:  rc.KeyPrefix,
		TTLSeconds: rc.TTLSeconds,
	}
	redisCache, err := memory.NewRedisCache(ctx, cacheCfg)
	if err != nil {
		logging.Warnf("Memory: Redis cache disabled (connection failed: %v)", err)
		return store
	}
	return memory.NewCachingStore(store, redisCache, backend)
}

// createMilvusMemoryStore creates a MilvusStore backend.
func createMilvusMemoryStore(cfg *config.RouterConfig, sets ...*embedding.Set) (memory.Store, error) {
	milvusAddress := cfg.Memory.Milvus.Address
	if milvusAddress == "" {
		milvusAddress = "localhost:19530"
	}

	collectionName := cfg.Memory.Milvus.Collection
	if collectionName == "" {
		collectionName = "agentic_memory"
	}

	embeddingConfig := &memory.EmbeddingConfig{
		Model:     memory.EmbeddingModelType(detectMemoryEmbeddingModel(cfg)),
		Dimension: cfg.Memory.Milvus.Dimension,
	}
	if len(sets) > 0 && sets[0] != nil {
		provider, err := sets[0].Get(string(embeddingConfig.Model), 0, 0)
		if err != nil {
			return nil, err
		}
		embeddingConfig.Provider = provider
	}

	logging.Infof("Memory: connecting to Milvus at %s, collection=%s, embedding=%s", milvusAddress, collectionName, embeddingConfig.Model)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	milvusClient, err := milvuslifecycle.ConnectGRPC(ctx, milvusAddress, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to create Milvus client: %w", err)
	}

	state, err := milvusClient.CheckHealth(ctx)
	if err != nil {
		_ = milvusClient.Close()
		return nil, fmt.Errorf("failed to check Milvus connection: %w", err)
	}
	if state == nil || !state.IsHealthy {
		_ = milvusClient.Close()
		return nil, fmt.Errorf("milvus connection is not healthy")
	}

	store, err := memory.NewMilvusStore(memory.MilvusStoreOptions{
		Client:          milvusClient,
		CollectionName:  collectionName,
		Config:          cfg.Memory,
		Enabled:         true,
		EmbeddingConfig: embeddingConfig,
	})
	if err != nil {
		_ = milvusClient.Close()
		return nil, fmt.Errorf("failed to create memory store: %w", err)
	}

	logging.Infof("Memory store initialized: backend=milvus, address=%s, collection=%s, embedding=%s",
		milvusAddress, collectionName, embeddingConfig.Model)

	return store, nil
}

// normalizeValkeyDimension sets vc.Dimension to the model's default if not explicitly configured.
func normalizeValkeyDimension(vc *config.MemoryValkeyConfig, model memory.EmbeddingModelType) {
	if vc.Dimension > 0 {
		return
	}
	switch model {
	case memory.EmbeddingModelMMBERT:
		vc.Dimension = 256
	default:
		vc.Dimension = 384
	}
	logging.Infof("Memory: Valkey dimension not set, defaulting to %d for model %s", vc.Dimension, model)
}

// createQdrantMemoryStore creates a QdrantStore backend.
func createQdrantMemoryStore(cfg *config.RouterConfig, sets ...*embedding.Set) (memory.Store, error) {
	qc := cfg.Memory.Qdrant
	if qc == nil {
		return nil, fmt.Errorf("memory.qdrant configuration is required when backend is 'qdrant'")
	}

	host := qc.Host
	if host == "" {
		host = "localhost"
	}
	port := qc.Port
	if port <= 0 {
		port = 6334
	}
	timeout := qc.ConnectTimeout
	if timeout <= 0 {
		timeout = 10
	}

	embeddingModel := memory.EmbeddingModelType(detectMemoryEmbeddingModel(cfg))

	embeddingConfig := &memory.EmbeddingConfig{
		Model:     embeddingModel,
		Dimension: qc.Dimension,
	}
	if len(sets) > 0 && sets[0] != nil {
		provider, err := sets[0].Get(string(embeddingConfig.Model), 0, 0)
		if err != nil {
			return nil, err
		}
		embeddingConfig.Provider = provider
	}

	logging.Infof("Memory: connecting to Qdrant at %s:%d, collection=%s, embedding=%s",
		host, port, qc.Collection, embeddingModel)

	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   host,
		Port:   port,
		APIKey: qc.APIKey,
		UseTLS: qc.UseTLS,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create qdrant client: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()
	if _, connErr := client.ListCollections(ctx); connErr != nil {
		_ = client.Close()
		return nil, fmt.Errorf("qdrant connection check failed: %w", connErr)
	}

	store, err := memory.NewQdrantStore(memory.QdrantStoreOptions{
		Client:          client,
		Config:          cfg.Memory,
		QdrantConfig:    qc,
		Enabled:         true,
		EmbeddingConfig: embeddingConfig,
	})
	if err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("failed to create qdrant memory store: %w", err)
	}

	logging.Infof("Memory store initialized: backend=qdrant, address=%s:%d, collection=%s, embedding=%s",
		host, port, qc.Collection, embeddingModel)

	return store, nil
}

func detectMemoryEmbeddingModel(cfg *config.RouterConfig) string {
	embeddingModels := cfg.EmbeddingModels
	embeddingModel := cfg.Memory.EmbeddingModel
	if embeddingModel != "" {
		return embeddingModel
	}

	switch {
	case embeddingModels.BertModelPath != "":
		logging.Infof("Memory: Auto-selected bert from embedding_models config (384-dim, recommended for memory)")
		return "bert"
	case embeddingModels.MmBertModelPath != "":
		logging.Infof("Memory: Auto-selected mmbert from embedding_models config")
		return "mmbert"
	case embeddingModels.MultiModalModelPath != "":
		logging.Infof("Memory: Auto-selected multimodal from embedding_models config")
		return "multimodal"
	case embeddingModels.Qwen3ModelPath != "":
		logging.Infof("Memory: Auto-selected qwen3 from embedding_models config")
		return "qwen3"
	case embeddingModels.GemmaModelPath != "":
		logging.Infof("Memory: Auto-selected gemma from embedding_models config")
		return "gemma"
	default:
		logging.Warnf("Memory: No embedding models configured, bert will be used but may fail without bert_model_path")
		return "bert"
	}
}
