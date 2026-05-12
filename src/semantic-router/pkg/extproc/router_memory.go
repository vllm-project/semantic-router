package extproc

import (
	"context"
	"fmt"
	"time"

	glide "github.com/valkey-io/valkey-glide/go/v2"
	glideconfig "github.com/valkey-io/valkey-glide/go/v2/config"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
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

	for _, decision := range cfg.Decisions {
		if decision.HasPlugin("memory") {
			logging.Infof("Memory auto-enabled: decision '%s' uses memory plugin", decision.Name)
			return true
		}
	}

	return false
}

// createMemoryStore creates a memory store based on configuration.
// Switches on cfg.Memory.Backend: "valkey" creates a ValkeyStore, "milvus" (or empty) creates a MilvusStore.
func createMemoryStore(cfg *config.RouterConfig) (memory.Store, error) {
	backend := cfg.Memory.Backend
	if backend == "" {
		backend = "milvus"
	}

	var store memory.Store
	var err error

	switch backend {
	case "valkey":
		store, err = createValkeyMemoryStore(cfg)
	case "milvus":
		store, err = createMilvusMemoryStore(cfg)
	default:
		return nil, fmt.Errorf("unsupported memory backend: %q (supported: milvus, valkey)", backend)
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
func createMilvusMemoryStore(cfg *config.RouterConfig) (memory.Store, error) {
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

// createValkeyMemoryStore creates a ValkeyStore backend.
func createValkeyMemoryStore(cfg *config.RouterConfig) (memory.Store, error) {
	vc := cfg.Memory.Valkey
	if vc == nil {
		return nil, fmt.Errorf("memory.valkey configuration is required when backend is 'valkey'")
	}

	host := vc.Host
	if host == "" {
		host = "localhost"
	}
	port := vc.Port
	if port <= 0 {
		port = 6379
	}

	embeddingModel := memory.EmbeddingModelType(detectMemoryEmbeddingModel(cfg))
	normalizeValkeyDimension(vc, embeddingModel)

	embeddingConfig := &memory.EmbeddingConfig{
		Model:     embeddingModel,
		Dimension: vc.Dimension,
	}

	logging.Infof("Memory: connecting to Valkey at %s:%d, embedding=%s", host, port, embeddingConfig.Model)

	clientConfig, err := buildValkeyClientConfig(vc, host, port)
	if err != nil {
		return nil, err
	}

	valkeyClient, err := glide.NewClient(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Valkey client: %w", err)
	}

	store, err := memory.NewValkeyStore(memory.ValkeyStoreOptions{
		Client:          valkeyClient,
		Config:          cfg.Memory,
		ValkeyConfig:    vc,
		Enabled:         true,
		EmbeddingConfig: embeddingConfig,
	})
	if err != nil {
		valkeyClient.Close()
		return nil, fmt.Errorf("failed to create Valkey memory store: %w", err)
	}

	logging.Infof("Memory store initialized: backend=valkey, address=%s:%d, embedding=%s",
		host, port, embeddingConfig.Model)

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

// buildValkeyClientConfig constructs the valkey-glide client configuration.
func buildValkeyClientConfig(vc *config.MemoryValkeyConfig, host string, port int) (*glideconfig.ClientConfiguration, error) {
	clientConfig := glideconfig.NewClientConfiguration().
		WithAddress(&glideconfig.NodeAddress{
			Host: host,
			Port: port,
		})

	if vc.Password != "" {
		clientConfig = clientConfig.WithCredentials(
			glideconfig.NewServerCredentials("", vc.Password),
		)
	}

	if vc.Database != 0 {
		clientConfig = clientConfig.WithDatabaseId(vc.Database)
	}

	if vc.Timeout > 0 {
		timeout := time.Duration(vc.Timeout) * time.Second
		clientConfig = clientConfig.WithRequestTimeout(timeout)
	}

	if vc.TLSEnabled {
		tlsCfg, tlsErr := buildValkeyTLSConfig(vc)
		if tlsErr != nil {
			return nil, tlsErr
		}
		clientConfig = clientConfig.WithUseTLS(true).
			WithAdvancedConfiguration(
				glideconfig.NewAdvancedClientConfiguration().WithTlsConfiguration(tlsCfg),
			)
		logging.Infof("Memory: Valkey TLS enabled (ca_path=%q, insecure_skip_verify=%v)", vc.TLSCAPath, vc.TLSInsecureSkipVerify)
	}

	return clientConfig, nil
}

// buildValkeyTLSConfig constructs a glide TLS configuration from the Valkey config.
func buildValkeyTLSConfig(vc *config.MemoryValkeyConfig) (*glideconfig.TlsConfiguration, error) {
	tlsConfig := glideconfig.NewTlsConfiguration()
	if vc.TLSCAPath != "" {
		caCert, err := glideconfig.LoadRootCertificatesFromFile(vc.TLSCAPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load TLS CA certificate from %s: %w", vc.TLSCAPath, err)
		}
		tlsConfig = tlsConfig.WithRootCertificates(caCert)
	}
	if vc.TLSInsecureSkipVerify {
		tlsConfig = tlsConfig.WithInsecureTLS(true)
		logging.Warnf("Memory: Valkey TLS certificate verification is DISABLED — do not use in production")
	}
	return tlsConfig, nil
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
