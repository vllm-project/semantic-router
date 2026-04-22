package cache

import (
	"context"
	"crypto/md5"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	glide "github.com/valkey-io/valkey-glide/go/v2"
	"github.com/valkey-io/valkey-glide/go/v2/config"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	valkeyutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/valkey"
)

// ValkeyCache provides a scalable semantic cache implementation using Valkey with vector search
type ValkeyCache struct {
	SimilarityTracker   // embedded — provides LastSimilarity()
	client              *glide.Client
	config              *routerconfig.ValkeyConfig
	indexName           string
	similarityThreshold float32
	ttlSeconds          int
	enabled             bool
	hitCount            int64
	missCount           int64
	lastCleanupTime     *time.Time
	mu                  sync.RWMutex
	embeddingModel      string // "bert", "qwen3", "gemma", "mmbert", or "multimodal"
}

// ValkeyCacheOptions contains configuration parameters for Valkey cache initialization
type ValkeyCacheOptions struct {
	SimilarityThreshold float32
	TTLSeconds          int
	Enabled             bool
	Config              *routerconfig.ValkeyConfig
	EmbeddingModel      string
}

// NewValkeyCache initializes a new Valkey-backed semantic cache instance
func NewValkeyCache(options ValkeyCacheOptions) (*ValkeyCache, error) {
	if !options.Enabled {
		logging.Debugf("ValkeyCache: disabled, returning stub")
		return &ValkeyCache{
			enabled: false,
		}, nil
	}

	if options.Config == nil {
		return nil, fmt.Errorf("valkey config is required")
	}

	valkeyConfig := options.Config
	// Normalize metric type to uppercase so configs using e.g. "cosine" match the
	// expected enum values (COSINE, IP, L2) without triggering fallback warnings.
	valkeyConfig.Index.VectorField.MetricType = strings.ToUpper(valkeyConfig.Index.VectorField.MetricType)
	logging.Debugf("ValkeyCache: config loaded - host=%s:%d, index=%s, dimension=auto-detect",
		valkeyConfig.Connection.Host, valkeyConfig.Connection.Port, valkeyConfig.Index.Name)

	resolvedHost := normalizeLocalHostForContainerRuntimes(valkeyConfig.Connection.Host)
	logging.Debugf("ValkeyCache: connecting to Valkey at %s:%d (configured host=%s)",
		resolvedHost, valkeyConfig.Connection.Port, valkeyConfig.Connection.Host)

	clientConfig := config.NewClientConfiguration().
		WithAddress(&config.NodeAddress{
			Host: resolvedHost,
			Port: valkeyConfig.Connection.Port,
		})

	if valkeyConfig.Connection.Password != "" {
		clientConfig = clientConfig.WithCredentials(
			config.NewServerCredentials("", valkeyConfig.Connection.Password),
		)
	}

	if valkeyConfig.Connection.Database != 0 {
		clientConfig = clientConfig.WithDatabaseId(valkeyConfig.Connection.Database)
	}

	if valkeyConfig.Connection.Timeout > 0 {
		timeout := time.Duration(valkeyConfig.Connection.Timeout) * time.Second
		clientConfig = clientConfig.WithRequestTimeout(timeout)
	}

	valkeyClient, err := glide.NewClient(clientConfig)
	if err != nil {
		logging.Debugf("ValkeyCache: failed to create client: %v", err)
		return nil, fmt.Errorf("failed to create Valkey client: %w", err)
	}

	embeddingModel := options.EmbeddingModel
	if embeddingModel == "" {
		embeddingModel = "bert"
	}

	cache := &ValkeyCache{
		client:              valkeyClient,
		config:              valkeyConfig,
		indexName:           valkeyConfig.Index.Name,
		similarityThreshold: options.SimilarityThreshold,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
		embeddingModel:      embeddingModel,
	}

	if err := cache.CheckConnection(); err != nil {
		logging.Debugf("ValkeyCache: failed to connect: %v", err)
		return nil, err
	}
	logging.Debugf("ValkeyCache: successfully connected to Valkey")

	logging.Debugf("ValkeyCache: initializing index '%s'", valkeyConfig.Index.Name)
	if err := cache.initializeSearchIndex(); err != nil {
		logging.Debugf("ValkeyCache: initialization failed: %v", err)
		valkeyClient.Close()
		return nil, err
	}
	logging.Debugf("ValkeyCache: initialization complete")

	return cache, nil
}

// initializeSearchIndex runs the valkey-search module version pre-check and
// then sets up the FT index.
func (c *ValkeyCache) initializeSearchIndex() error {
	versionCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := valkeyutil.EnsureSearchModuleVersion(versionCtx, c.client, valkeyutil.SearchModuleMinVersion); err != nil {
		return err
	}
	if err := c.initializeIndex(); err != nil {
		return fmt.Errorf("failed to initialize index: %w", err)
	}
	return nil
}

// initializeIndex sets up the Valkey index for vector search
func (c *ValkeyCache) initializeIndex() error {
	ctx := context.Background()

	_, err := c.client.CustomCommand(ctx, []string{"FT.INFO", c.indexName})
	indexExists := err == nil

	if c.config.Development.DropIndexOnStartup && indexExists {
		_, err := c.client.CustomCommand(ctx, []string{"FT.DROPINDEX", c.indexName})
		if err != nil {
			logging.Debugf("ValkeyCache: failed to drop index: %v", err)
			return fmt.Errorf("failed to drop index: %w", err)
		}
		indexExists = false
		logging.Debugf("ValkeyCache: dropped existing index '%s' for development", c.indexName)
		logging.LogEvent("index_dropped", map[string]interface{}{
			"backend": "valkey",
			"index":   c.indexName,
			"reason":  "development_mode",
		})
	}

	if !indexExists {
		if !c.config.Development.AutoCreateIndex {
			return fmt.Errorf("index %s does not exist and auto-creation is disabled", c.indexName)
		}

		if err := c.createIndex(); err != nil {
			logging.Debugf("ValkeyCache: failed to create index: %v", err)
			return fmt.Errorf("failed to create index: %w", err)
		}
		logging.Debugf("ValkeyCache: created new index '%s' with dimension %d",
			c.indexName, c.config.Index.VectorField.Dimension)
		logging.LogEvent("index_created", map[string]interface{}{
			"backend":   "valkey",
			"index":     c.indexName,
			"dimension": c.config.Index.VectorField.Dimension,
		})
	}

	return nil
}

// getEmbedding generates an embedding based on the configured embedding model
func (c *ValkeyCache) getEmbedding(text string) ([]float32, error) {
	modelName := strings.ToLower(strings.TrimSpace(c.embeddingModel))

	switch modelName {
	case "qwen3":
		// Use GetEmbeddingBatched for Qwen3 with batching support
		output, err := candle_binding.GetEmbeddingBatched(text, modelName, 0)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "gemma":
		// Use GetEmbeddingWithModelType for Gemma
		output, err := candle_binding.GetEmbeddingWithModelType(text, modelName, 0)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "mmbert":
		// Use GetEmbedding2DMatryoshka for mmBERT
		output, err := candle_binding.GetEmbedding2DMatryoshka(text, modelName, 0, 0)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "multimodal":
		// Use multimodal text encoder branch (384-dim default)
		output, err := candle_binding.GetEmbeddingWithModelType(text, modelName, 384)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "bert", "":
		// Use traditional GetEmbedding for BERT (default)
		return candle_binding.GetEmbedding(text, 0)
	default:
		return nil, fmt.Errorf("unsupported embedding model: %s (must be 'bert', 'qwen3', 'gemma', 'mmbert', or 'multimodal')", c.embeddingModel)
	}
}

// createIndex builds the Valkey index with the appropriate schema
func (c *ValkeyCache) createIndex() error {
	ctx := context.Background()

	testEmbedding, err := c.getEmbedding("test")
	if err != nil {
		return fmt.Errorf("failed to detect embedding dimension: %w", err)
	}
	actualDimension := len(testEmbedding)

	logging.Debugf("ValkeyCache.createIndex: auto-detected embedding dimension: %d", actualDimension)

	var distanceMetric string
	switch c.config.Index.VectorField.MetricType {
	case "L2":
		distanceMetric = "L2"
	case "IP":
		distanceMetric = "IP"
	case "COSINE":
		distanceMetric = "COSINE"
	default:
		logging.Warnf("ValkeyCache: unknown metric type '%s', defaulting to COSINE", c.config.Index.VectorField.MetricType)
		distanceMetric = "COSINE"
	}

	var createCmd []string
	if c.config.Index.IndexType == "HNSW" {
		createCmd = []string{
			"FT.CREATE", c.indexName,
			"ON", "HASH",
			"PREFIX", "1", c.config.Index.Prefix,
			"SCHEMA",
			"request_id", "TAG",
			"model", "TAG",
			"query", "TEXT",
			c.config.Index.VectorField.Name, "VECTOR", "HNSW", "10",
			"TYPE", "FLOAT32",
			"DIM", fmt.Sprintf("%d", actualDimension),
			"DISTANCE_METRIC", distanceMetric,
			"M", fmt.Sprintf("%d", c.config.Index.Params.M),
			"EF_CONSTRUCTION", fmt.Sprintf("%d", c.config.Index.Params.EfConstruction),
			"timestamp", "NUMERIC",
		}
	} else {
		createCmd = []string{
			"FT.CREATE", c.indexName,
			"ON", "HASH",
			"PREFIX", "1", c.config.Index.Prefix,
			"SCHEMA",
			"request_id", "TAG",
			"model", "TAG",
			"query", "TEXT",
			c.config.Index.VectorField.Name, "VECTOR", "FLAT", "6",
			"TYPE", "FLOAT32",
			"DIM", fmt.Sprintf("%d", actualDimension),
			"DISTANCE_METRIC", distanceMetric,
			"timestamp", "NUMERIC",
		}
	}

	_, err = c.client.CustomCommand(ctx, createCmd)
	if err != nil {
		return fmt.Errorf("failed to create Valkey index: %w", err)
	}

	return nil
}

// IsEnabled returns the current cache activation status
func (c *ValkeyCache) IsEnabled() bool {
	return c.enabled
}

// CheckConnection verifies the Valkey connection is healthy
func (c *ValkeyCache) CheckConnection() error {
	if !c.enabled {
		return nil
	}

	if c.client == nil {
		return fmt.Errorf("valkey client is not initialized")
	}

	ctx := context.Background()
	if c.config != nil && c.config.Connection.Timeout > 0 {
		timeout := time.Duration(c.config.Connection.Timeout) * time.Second
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	_, err := c.client.Ping(ctx)
	if err != nil {
		return fmt.Errorf("valkey connection check failed: %w", err)
	}

	return nil
}

// AddPendingRequest stores a request that is awaiting its response
func (c *ValkeyCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	if ttlSeconds == 0 {
		logging.Debugf("ValkeyCache.AddPendingRequest: skipping cache (ttl_seconds=0)")
		return nil
	}

	err := c.addEntry("", requestID, model, query, requestBody, nil, ttlSeconds)

	if err != nil {
		metrics.RecordCacheOperation("valkey", "add_pending", "error", time.Since(start).Seconds())
	} else {
		metrics.RecordCacheOperation("valkey", "add_pending", "success", time.Since(start).Seconds())
	}

	return err
}

func (c *ValkeyCache) UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	logging.Debugf("ValkeyCache.UpdateWithResponse: updating pending entry (request_id: %s, response_size: %d, ttl_seconds=%d)",
		requestID, len(responseBody), ttlSeconds)

	ctx := context.Background()

	query := fmt.Sprintf("@request_id:{%s}", escapeTagValue(requestID))
	logging.Debugf("UpdateWithResponse: searching with TAG query: %s", query)

	searchCmd := []string{
		"FT.SEARCH", c.indexName, query,
		"RETURN", "3", "model", "query", "request_body",
		"LIMIT", "0", "1",
	}

	results, err := c.client.CustomCommand(ctx, searchCmd)
	if err != nil {
		logging.Infof("ValkeyCache.UpdateWithResponse: TAG search failed: %v", err)
		metrics.RecordCacheOperation("valkey", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to search pending entry: %w", err)
	}

	logging.Debugf("UpdateWithResponse: search results type=%T, value=%v", results, results)

	entry, err := parsePendingSearchResult(results, requestID, c.config.Index.Prefix)
	if err != nil {
		metrics.RecordCacheOperation("valkey", "update_response", "error", time.Since(start).Seconds())
		return err
	}

	logging.Debugf("ValkeyCache.UpdateWithResponse: found pending entry, updating (id: %s, model: %s)", entry.docID, entry.model)

	// Update the document with response body and TTL
	err = c.addEntry(entry.docID, requestID, entry.model, entry.query, []byte(entry.requestBodyStr), responseBody, ttlSeconds)
	if err != nil {
		metrics.RecordCacheOperation("valkey", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to update entry: %w", err)
	}

	logging.Debugf("ValkeyCache.UpdateWithResponse: successfully updated entry with response")
	metrics.RecordCacheOperation("valkey", "update_response", "success", time.Since(start).Seconds())

	return nil
}

// AddEntry stores a complete request-response pair in the cache
func (c *ValkeyCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	if ttlSeconds == 0 {
		logging.Debugf("ValkeyCache.AddEntry: skipping cache (ttl_seconds=0)")
		return nil
	}

	err := c.addEntry("", requestID, model, query, requestBody, responseBody, ttlSeconds)

	if err != nil {
		metrics.RecordCacheOperation("valkey", "add_entry", "error", time.Since(start).Seconds())
	} else {
		metrics.RecordCacheOperation("valkey", "add_entry", "success", time.Since(start).Seconds())
	}

	return err
}

// addEntry handles the internal logic for storing entries in Valkey
func (c *ValkeyCache) addEntry(id string, requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	effectiveTTL := ttlSeconds
	if ttlSeconds == -1 {
		effectiveTTL = c.ttlSeconds
	}

	embedding, err := c.getEmbedding(query)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	if id == "" {
		id = fmt.Sprintf("%x", md5.Sum(fmt.Appendf(nil, "%s_%s_%d", model, query, time.Now().UnixNano())))
	}

	ctx := context.Background()

	embeddingBytes := floatsToBytes(embedding)

	var docKey string
	if strings.HasPrefix(id, c.config.Index.Prefix) {
		docKey = id // Already has prefix, use as-is
	} else {
		docKey = c.config.Index.Prefix + id // Add prefix
	}

	hsetCmd := []string{
		"HSET", docKey,
		"request_id", requestID,
		"model", model,
		"query", query,
		"request_body", string(requestBody),
		"response_body", string(responseBody),
		c.config.Index.VectorField.Name, string(embeddingBytes),
		"timestamp", fmt.Sprintf("%d", time.Now().Unix()),
		"ttl_seconds", fmt.Sprintf("%d", effectiveTTL),
	}

	_, err = c.client.CustomCommand(ctx, hsetCmd)
	if err != nil {
		logging.Debugf("ValkeyCache.addEntry: HSET failed: %v", err)
		return fmt.Errorf("failed to store cache entry: %w", err)
	}

	if effectiveTTL > 0 {
		expireCmd := []string{"EXPIRE", docKey, fmt.Sprintf("%d", effectiveTTL)}
		_, _ = c.client.CustomCommand(ctx, expireCmd)
	}

	logging.Debugf("ValkeyCache.addEntry: successfully added entry to Valkey (key: %s, embedding_dim: %d, request_size: %d, response_size: %d, ttl=%d)",
		docKey, len(embedding), len(requestBody), len(responseBody), effectiveTTL)
	logging.LogEvent("cache_entry_added", map[string]interface{}{
		"backend":             "valkey",
		"index":               c.indexName,
		"request_id":          requestID,
		"query":               query,
		"model":               model,
		"embedding_dimension": len(embedding),
	})
	return nil
}

// FindSimilar searches for semantically similar cached requests
func (c *ValkeyCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

// buildKNNSearchCmd constructs the FT.SEARCH command for a KNN vector similarity query.
func (c *ValkeyCache) buildKNNSearchCmd(embeddingBytes []byte) []string {
	knnQuery := fmt.Sprintf("*=>[KNN %d @%s $vec AS vector_distance]",
		c.config.Search.TopK, c.config.Index.VectorField.Name)

	cmd := []string{
		"FT.SEARCH", c.indexName, knnQuery,
		"RETURN", "2", "vector_distance", "response_body",
		"DIALECT", "2",
		"PARAMS", "2", "vec",
	}
	return append(cmd, string(embeddingBytes))
}

// recordCacheMiss increments the miss counter and records the metric.
func (c *ValkeyCache) recordCacheMiss(status string, elapsed time.Duration) {
	atomic.AddInt64(&c.missCount, 1)
	metrics.RecordCacheOperation("valkey", "find_similar", status, elapsed.Seconds())
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *ValkeyCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	start := time.Now()

	if !c.enabled {
		return nil, false, nil
	}

	queryEmbedding, err := c.getEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("valkey", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	ctx := context.Background()

	embeddingBytes := floatsToBytes(queryEmbedding)
	searchCmd := c.buildKNNSearchCmd(embeddingBytes)

	searchResult, err := c.client.CustomCommand(ctx, searchCmd)
	if err != nil {
		logging.Debugf("ValkeyCache.FindSimilarWithThreshold: search failed: %v", err)
		c.recordCacheMiss("error", time.Since(start))
		return nil, false, nil
	}

	match := parseBestMatch(searchResult)
	if match == nil {
		c.recordCacheMiss("miss", time.Since(start))
		return nil, false, nil
	}

	similarity := distanceToSimilarity(c.config.Index.VectorField.MetricType, match.distance)
	c.StoreSimilarity(similarity)

	if similarity < threshold {
		logging.Debugf("ValkeyCache.FindSimilarWithThreshold: cache miss - similarity %.4f below threshold %.4f",
			similarity, threshold)
		logging.LogEvent("cache_miss", map[string]interface{}{
			"backend":         "valkey",
			"best_similarity": similarity,
			"threshold":       threshold,
			"model":           model,
			"index":           c.indexName,
		})
		c.recordCacheMiss("miss", time.Since(start))
		return nil, false, nil
	}

	responseBody := extractResponseBody(match)
	if responseBody == nil {
		c.recordCacheMiss("error", time.Since(start))
		return nil, false, nil
	}

	atomic.AddInt64(&c.hitCount, 1)
	logging.Debugf("ValkeyCache.FindSimilarWithThreshold: cache hit - similarity=%.4f, response_size=%d bytes",
		similarity, len(responseBody))
	logging.LogEvent("cache_hit", map[string]interface{}{
		"backend":    "valkey",
		"similarity": similarity,
		"threshold":  threshold,
		"model":      model,
		"index":      c.indexName,
	})
	metrics.RecordCacheOperation("valkey", "find_similar", "hit", time.Since(start).Seconds())
	return responseBody, true, nil
}

// Close releases all resources held by the cache
func (c *ValkeyCache) Close() error {
	if c.client != nil {
		c.client.Close()
	}
	return nil
}

// getIndexEntryCount retrieves the number of indexed documents from Valkey FT.INFO.
func (c *ValkeyCache) getIndexEntryCount() int {
	if !c.enabled || c.client == nil {
		return 0
	}

	ctx := context.Background()
	info, err := c.client.CustomCommand(ctx, []string{"FT.INFO", c.indexName})
	if err != nil {
		return 0
	}

	infoMap, ok := info.(map[string]interface{})
	if !ok {
		return 0
	}

	numDocs, exists := infoMap["num_docs"]
	if !exists {
		return 0
	}

	count, ok := numDocs.(int64)
	if !ok {
		return 0
	}

	return int(count)
}

// GetStats provides current cache performance metrics
func (c *ValkeyCache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	hits := atomic.LoadInt64(&c.hitCount)
	misses := atomic.LoadInt64(&c.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	cacheStats := CacheStats{
		TotalEntries: c.getIndexEntryCount(),
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}

	if c.lastCleanupTime != nil {
		cacheStats.LastCleanupTime = c.lastCleanupTime
	}

	return cacheStats
}
