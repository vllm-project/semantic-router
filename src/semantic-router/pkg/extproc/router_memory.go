package extproc

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func createMemoryRuntime(cfg *config.RouterConfig) (memory.Store, *memory.MemoryExtractor) {
	if !isMemoryEnabled(cfg) {
		return nil, nil
	}

	memoryStore, err := createMemoryStore(cfg)
	if err != nil {
		logging.Warnf("Failed to create memory store: %v, Memory will be disabled", err)
		return nil, nil
	}

	memory.SetGlobalMemoryStore(memoryStore)
	if rc := cfg.Memory.RedisCache; rc != nil && rc.Enabled && rc.Address != "" {
		logging.Infof("Memory enabled with Milvus backend and Redis hot cache")
	} else {
		logging.Infof("Memory enabled with Milvus backend")
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

	for _, decision := range cfg.Decisions {
		if decision.HasPlugin("memory") {
			logging.Infof("Memory auto-enabled: decision '%s' uses memory plugin", decision.Name)
			return true
		}
	}

	return false
}

// createMemoryStore creates a memory store based on configuration.
func createMemoryStore(cfg *config.RouterConfig) (memory.Store, error) {
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

	logging.Infof("Memory: Connecting to Milvus at %s, collection=%s", milvusAddress, collectionName)
	logging.Infof("Memory: Using embedding model=%s", embeddingConfig.Model)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	milvusClient, err := client.NewGrpcClient(ctx, milvusAddress)
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

	logging.Infof(
		"Memory store initialized: address=%s, collection=%s, embedding=%s",
		milvusAddress,
		collectionName,
		embeddingConfig.Model,
	)

	var result memory.Store = store
	if rc := cfg.Memory.RedisCache; rc != nil && rc.Enabled && rc.Address != "" {
		cacheCfg := &memory.RedisCacheConfig{
			Address:    rc.Address,
			Password:   rc.Password,
			DB:         rc.DB,
			KeyPrefix:  rc.KeyPrefix,
			TTLSeconds: rc.TTLSeconds,
		}
		redisCache, err := memory.NewRedisCache(ctx, cacheCfg)
		if err != nil {
			logging.Warnf("Memory: Redis cache disabled (connection failed: %v), using Milvus only", err)
		} else {
			result = memory.NewCachingStore(store, redisCache)
		}
	}

	return result, nil
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
