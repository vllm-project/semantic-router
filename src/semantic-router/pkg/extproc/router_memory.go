package extproc

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/qdrant/go-client/qdrant"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func createMemoryRuntime(cfg *config.RouterConfig, sets ...*embedding.Set) (memory.Store, *memory.MemoryExtractor, routerruntime.MemoryStatus, error) {
	if err := validateMemoryFilterAlgorithms(cfg); err != nil {
		return nil, nil, routerruntime.MemoryStatus{}, err
	}
	backend := cfg.Memory.Backend
	if backend == "" {
		backend = "milvus"
	}
	if !cfg.UsesMemory() {
		return nil, nil, routerruntime.MemoryStatus{
			Reason:  routerruntime.MemoryReasonDisabledByConfig,
			Backend: backend,
		}, nil
	}

	// publishRouterState publishes the store after the candidate commits.
	memoryStore, err := createMemoryStore(cfg, sets...)
	if err != nil {
		logging.Warnf("Failed to create memory store: %v, Memory will be disabled", err)
		return nil, nil, routerruntime.MemoryStatus{
			Reason:  routerruntime.MemoryReasonBackendUnavailable,
			Backend: backend,
		}, nil
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

	return memoryStore, memoryExtractor, routerruntime.MemoryStatus{
		Backend: backend,
	}, nil
}

// legacyMemoryFilterAlgorithms maps retired algorithm names to the filter they
// always ran as.
var legacyMemoryFilterAlgorithms = map[string]string{"recency_semantic": "heuristic"}

// validateMemoryFilterAlgorithms rejects reflection algorithms that no filter is
// registered for. The config package cannot import the registry, so the check
// runs here once at startup. A decision's effective algorithm is its own
// override or the global value, so checking both covers every decision.
func validateMemoryFilterAlgorithms(cfg *config.RouterConfig) error {
	if err := validateMemoryFilterAlgorithm(cfg.Memory.Reflection.Algorithm, "global.stores.memory.reflection.algorithm"); err != nil {
		return err
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		source := decision.GetContextConfig().Source(config.ContextSourceMemory)
		if source == nil || source.Reflection == nil {
			continue
		}
		field := fmt.Sprintf("routing.decisions[%s].plugins[context].sources[memory].reflection.algorithm", decision.Name)
		if err := validateMemoryFilterAlgorithm(source.Reflection.Algorithm, field); err != nil {
			return err
		}
	}
	return nil
}

func validateMemoryFilterAlgorithm(algorithm, field string) error {
	if algorithm == "" {
		return nil
	}
	registered := memory.RegisteredFilters()
	if slices.Contains(registered, algorithm) {
		return nil
	}
	slices.Sort(registered)
	if replacement, ok := legacyMemoryFilterAlgorithms[algorithm]; ok {
		return fmt.Errorf("%s %q is not a registered memory filter (registered: %s); use %q, which is how %q has always run",
			field, algorithm, strings.Join(registered, ", "), replacement, algorithm)
	}
	return fmt.Errorf("%s %q is not a registered memory filter (registered: %s)",
		field, algorithm, strings.Join(registered, ", "))
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

	embeddingConfig, err := memoryEmbeddingConfig(cfg, sets)
	if err != nil {
		return nil, err
	}
	dimension, err := memory.StorageDimension(cfg.Memory.Milvus.Dimension, *embeddingConfig)
	if err != nil {
		return nil, err
	}
	copied := *cfg
	copied.Memory.Milvus.Dimension = dimension
	cfg = &copied
	embeddingConfig.Dimension = dimension

	logging.Infof("Memory: connecting to Milvus at %s, collection=%s, embedding=%s", milvusAddress, collectionName, embeddingConfig.Deployment)

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
		milvusAddress, collectionName, embeddingConfig.Deployment)

	return store, nil
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

	embeddingConfig, err := memoryEmbeddingConfig(cfg, sets)
	if err != nil {
		return nil, err
	}
	dimension, err := memory.StorageDimension(qc.Dimension, *embeddingConfig)
	if err != nil {
		return nil, err
	}
	copied := *qc
	qc = &copied
	qc.Dimension = dimension
	embeddingConfig.Dimension = dimension

	logging.Infof("Memory: connecting to Qdrant at %s:%d, collection=%s, embedding=%s",
		host, port, qc.Collection, embeddingConfig.Deployment)

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
		host, port, qc.Collection, embeddingConfig.Deployment)

	return store, nil
}

// memoryEmbeddingConfig is the deployment memory embeds with and the width of
// its vectors, with that deployment's provider from the prepared set.
func memoryEmbeddingConfig(cfg *config.RouterConfig, sets []*embedding.Set) (*memory.EmbeddingConfig, error) {
	deployment, dimension := cfg.MemoryEmbedding()
	embeddingConfig := &memory.EmbeddingConfig{Deployment: deployment, Dimension: dimension}
	if len(sets) > 0 && sets[0] != nil {
		provider, err := sets[0].Get(deployment, 0, 0)
		if err != nil {
			return nil, err
		}
		embeddingConfig.Provider = provider
	}
	return embeddingConfig, nil
}
