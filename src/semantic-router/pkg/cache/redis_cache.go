package cache

import (
	"context"
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
	"sigs.k8s.io/yaml"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// RedisCache provides a scalable semantic cache implementation using Redis with vector search
type RedisCache struct {
	client              *redis.Client
	config              *config.RedisConfig
	indexName           string
	similarityThreshold float32
	ttlSeconds          int
	enabled             bool
	hitCount            int64
	missCount           int64
	lastCleanupTime     *time.Time
	mu                  sync.RWMutex
}

// RedisCacheOptions contains configuration parameters for Redis cache initialization
type RedisCacheOptions struct {
	SimilarityThreshold float32
	TTLSeconds          int
	Enabled             bool
	Config              *config.RedisConfig
	ConfigPath          string
}

// NewRedisCache initializes a new Redis-backed semantic cache instance
func NewRedisCache(options RedisCacheOptions) (*RedisCache, error) {
	if !options.Enabled {
		logging.Debugf("RedisCache: disabled, returning stub")
		return &RedisCache{
			enabled: false,
		}, nil
	}

	// (Fallback) Load Redis configuration from a separated configuration file
	var err error
	var redisConfig *config.RedisConfig
	if options.Config == nil {
		logging.Warnf("(Deprecated) RedisCache: loading config from %s", options.ConfigPath)
		redisConfig, err = loadRedisConfig(options.ConfigPath)
		if err != nil {
			logging.Debugf("RedisCache: failed to load config: %v", err)
			return nil, fmt.Errorf("failed to load Redis config: %w", err)
		}
	} else {
		redisConfig = options.Config
	}
	logging.Debugf("RedisCache: config loaded - host=%s:%d, index=%s, dimension=auto-detect",
		redisConfig.Connection.Host, redisConfig.Connection.Port, redisConfig.Index.Name)

	// Establish connection to Redis server
	logging.Debugf("RedisCache: connecting to Redis at %s:%d", redisConfig.Connection.Host, redisConfig.Connection.Port)

	redisClient := redis.NewClient(&redis.Options{
		Addr:     fmt.Sprintf("%s:%d", redisConfig.Connection.Host, redisConfig.Connection.Port),
		Password: redisConfig.Connection.Password,
		DB:       redisConfig.Connection.Database,
		Protocol: 2, // Use RESP2 protocol for compatibility
	})

	cache := &RedisCache{
		client:              redisClient,
		config:              redisConfig,
		indexName:           redisConfig.Index.Name,
		similarityThreshold: options.SimilarityThreshold,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
	}

	// Test connection using the new CheckConnection method
	if err := cache.CheckConnection(); err != nil {
		logging.Debugf("RedisCache: failed to connect: %v", err)
		return nil, err
	}
	logging.Debugf("RedisCache: successfully connected to Redis")

	// Set up the index for vector search
	logging.Debugf("RedisCache: initializing index '%s'", redisConfig.Index.Name)
	if err := cache.initializeIndex(); err != nil {
		logging.Debugf("RedisCache: failed to initialize index: %v", err)
		redisClient.Close()
		return nil, fmt.Errorf("failed to initialize index: %w", err)
	}
	logging.Debugf("RedisCache: initialization complete")

	return cache, nil
}

// loadRedisConfig reads and parses the Redis configuration from a file (Deprecated)
func loadRedisConfig(configPath string) (*config.RedisConfig, error) {
	if configPath == "" {
		return nil, fmt.Errorf("redis config path is required")
	}

	logging.Debugf("Loading Redis config from: %s", configPath)

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var redisConfig *config.RedisConfig
	if err := yaml.Unmarshal(data, &redisConfig); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	logging.Debugf("Redis config loaded: index=%s, dimension=%d, metric=%s",
		redisConfig.Index.Name, redisConfig.Index.VectorField.Dimension, redisConfig.Index.VectorField.MetricType)

	// Apply defaults
	if redisConfig.Index.VectorField.Name == "" {
		redisConfig.Index.VectorField.Name = "embedding"
		logging.Warnf("VectorField.Name not specified, using default: embedding")
	}
	if redisConfig.Index.VectorField.MetricType == "" {
		redisConfig.Index.VectorField.MetricType = "COSINE"
	}
	if redisConfig.Index.IndexType == "" {
		redisConfig.Index.IndexType = "HNSW"
	}
	if redisConfig.Index.Prefix == "" {
		redisConfig.Index.Prefix = "doc:"
	}
	// Validate index params for HNSW
	if redisConfig.Index.IndexType == "HNSW" {
		if redisConfig.Index.Params.M == 0 {
			redisConfig.Index.Params.M = 16
		}
		if redisConfig.Index.Params.EfConstruction == 0 {
			redisConfig.Index.Params.EfConstruction = 64
		}
	}
	if redisConfig.Search.TopK == 0 {
		redisConfig.Search.TopK = 1
	}

	return redisConfig, nil
}

// initializeIndex sets up the Redis index for vector search
func (c *RedisCache) initializeIndex() error {
	ctx := context.Background()

	// Check if index exists
	_, err := c.client.FTInfo(ctx, c.indexName).Result()
	indexExists := err == nil

	// Handle development mode index reset
	if c.config.Development.DropIndexOnStartup && indexExists {
		if err := c.client.FTDropIndexWithArgs(ctx, c.indexName, &redis.FTDropIndexOptions{
			DeleteDocs: true,
		}).Err(); err != nil {
			logging.Debugf("RedisCache: failed to drop index: %v", err)
			return fmt.Errorf("failed to drop index: %w", err)
		}
		indexExists = false
		logging.Debugf("RedisCache: dropped existing index '%s' for development", c.indexName)
		logging.LogEvent("index_dropped", map[string]interface{}{
			"backend": "redis",
			"index":   c.indexName,
			"reason":  "development_mode",
		})
	}

	// Create index if it doesn't exist
	if !indexExists {
		if !c.config.Development.AutoCreateIndex {
			return fmt.Errorf("index %s does not exist and auto-creation is disabled", c.indexName)
		}

		if err := c.createIndex(); err != nil {
			logging.Debugf("RedisCache: failed to create index: %v", err)
			return fmt.Errorf("failed to create index: %w", err)
		}
		logging.Debugf("RedisCache: created new index '%s' with dimension %d",
			c.indexName, c.config.Index.VectorField.Dimension)
		logging.LogEvent("index_created", map[string]interface{}{
			"backend":   "redis",
			"index":     c.indexName,
			"dimension": c.config.Index.VectorField.Dimension,
		})
	}

	return nil
}

// createIndex builds the Redis index with the appropriate schema
func (c *RedisCache) createIndex() error {
	ctx := context.Background()

	// Determine embedding dimension automatically
	testEmbedding, err := candle_binding.GetEmbedding("test", 0)
	if err != nil {
		return fmt.Errorf("failed to detect embedding dimension: %w", err)
	}
	actualDimension := len(testEmbedding)

	logging.Debugf("RedisCache.createIndex: auto-detected embedding dimension: %d", actualDimension)

	// Determine distance metric for Redis
	var distanceMetric string
	switch c.config.Index.VectorField.MetricType {
	case "L2":
		distanceMetric = "L2"
	case "IP":
		distanceMetric = "IP"
	case "COSINE":
		distanceMetric = "COSINE"
	default:
		logging.Warnf("RedisCache: unknown metric type '%s', defaulting to COSINE", c.config.Index.VectorField.MetricType)
		distanceMetric = "COSINE"
	}

	// Create vector field arguments based on index type
	var vectorArgs *redis.FTVectorArgs
	if c.config.Index.IndexType == "HNSW" {
		vectorArgs = &redis.FTVectorArgs{
			HNSWOptions: &redis.FTHNSWOptions{
				Type:                   "FLOAT32",
				Dim:                    actualDimension,
				DistanceMetric:         distanceMetric,
				MaxEdgesPerNode:        c.config.Index.Params.M,
				MaxAllowedEdgesPerNode: c.config.Index.Params.EfConstruction,
			},
		}
	} else {
		vectorArgs = &redis.FTVectorArgs{
			FlatOptions: &redis.FTFlatOptions{
				Type:           "FLOAT32",
				Dim:            actualDimension,
				DistanceMetric: distanceMetric,
			},
		}
	}

	// Create the index with proper schema
	_, err = c.client.FTCreate(ctx,
		c.indexName,
		&redis.FTCreateOptions{
			OnHash: true,
			Prefix: []interface{}{c.config.Index.Prefix},
		},
		&redis.FieldSchema{
			FieldName: "request_id",
			FieldType: redis.SearchFieldTypeText,
		},
		&redis.FieldSchema{
			FieldName: "model",
			FieldType: redis.SearchFieldTypeTag,
		},
		&redis.FieldSchema{
			FieldName: "query",
			FieldType: redis.SearchFieldTypeText,
		},
		&redis.FieldSchema{
			FieldName: "request_body",
			FieldType: redis.SearchFieldTypeText,
			NoIndex:   true, // Don't index large text fields
		},
		&redis.FieldSchema{
			FieldName: "response_body",
			FieldType: redis.SearchFieldTypeText,
			NoIndex:   true, // Don't index large text fields
		},
		&redis.FieldSchema{
			FieldName:  c.config.Index.VectorField.Name,
			FieldType:  redis.SearchFieldTypeVector,
			VectorArgs: vectorArgs,
		},
		&redis.FieldSchema{
			FieldName: "timestamp",
			FieldType: redis.SearchFieldTypeNumeric,
		},
	).Result()
	if err != nil {
		return fmt.Errorf("failed to create Redis index: %w", err)
	}

	return nil
}

// IsEnabled returns the current cache activation status
func (c *RedisCache) IsEnabled() bool {
	return c.enabled
}

// CheckConnection verifies the Redis connection is healthy
func (c *RedisCache) CheckConnection() error {
	if !c.enabled {
		return nil
	}

	if c.client == nil {
		return fmt.Errorf("redis client is not initialized")
	}

	ctx := context.Background()
	if c.config != nil && c.config.Connection.Timeout > 0 {
		timeout := time.Duration(c.config.Connection.Timeout) * time.Second
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	if err := c.client.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("redis connection check failed: %w", err)
	}

	return nil
}

// AddPendingRequest stores a request that is awaiting its response
func (c *RedisCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Handle TTL=0: skip caching entirely
	if ttlSeconds == 0 {
		logging.Debugf("RedisCache.AddPendingRequest: skipping cache (ttl_seconds=0)")
		return nil
	}

	// Store incomplete entry for later completion with response
	err := c.addEntry("", requestID, model, query, requestBody, nil, ttlSeconds)

	if err != nil {
		metrics.RecordCacheOperation("redis", "add_pending", "error", time.Since(start).Seconds())
	} else {
		metrics.RecordCacheOperation("redis", "add_pending", "success", time.Since(start).Seconds())
	}

	return err
}

// UpdateWithResponse completes a pending request by adding the response
func (c *RedisCache) UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	logging.Debugf("RedisCache.UpdateWithResponse: updating pending entry (request_id: %s, response_size: %d, ttl_seconds=%d)",
		requestID, len(responseBody), ttlSeconds)

	// Find the pending entry by request_id
	ctx := context.Background()

	// Search for documents with matching request_id using TEXT field syntax (exact match with quotes)
	// TAG syntax with {} doesn't work well with UUIDs containing hyphens
	query := fmt.Sprintf("@request_id:\"%s\"", requestID)
	logging.Infof("UpdateWithResponse: searching with query: %s", query)

	results, err := c.client.FTSearchWithArgs(ctx,
		c.indexName,
		query,
		&redis.FTSearchOptions{
			Return: []redis.FTSearchReturn{
				{FieldName: "model"},
				{FieldName: "query"},
				{FieldName: "request_body"},
			},
			LimitOffset: 0,
			Limit:       1,
		},
	).Result()
	if err != nil {
		logging.Infof("RedisCache.UpdateWithResponse: search failed with query '%s': %v", query, err)
		metrics.RecordCacheOperation("redis", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to search pending entry: %w", err)
	}

	if results.Total == 0 {
		logging.Infof("RedisCache.UpdateWithResponse: no pending entry found with request_id=%s", requestID)
		metrics.RecordCacheOperation("redis", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("no pending entry found")
	}

	logging.Infof("UpdateWithResponse: found %d result(s) for request_id=%s", results.Total, requestID)

	doc := results.Docs[0]
	model := fmt.Sprint(doc.Fields["model"])
	queryStr := fmt.Sprint(doc.Fields["query"])
	requestBodyStr := fmt.Sprint(doc.Fields["request_body"])

	// Extract document ID from the result
	docID := doc.ID

	logging.Debugf("RedisCache.UpdateWithResponse: found pending entry, updating (id: %s, model: %s)", docID, model)

	// Update the document with response body and TTL
	err = c.addEntry(docID, requestID, model, queryStr, []byte(requestBodyStr), responseBody, ttlSeconds)
	if err != nil {
		metrics.RecordCacheOperation("redis", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to update entry: %w", err)
	}

	logging.Debugf("RedisCache.UpdateWithResponse: successfully updated entry with response")
	metrics.RecordCacheOperation("redis", "update_response", "success", time.Since(start).Seconds())

	return nil
}

// AddEntry stores a complete request-response pair in the cache
func (c *RedisCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Handle TTL=0: skip caching entirely
	if ttlSeconds == 0 {
		logging.Debugf("RedisCache.AddEntry: skipping cache (ttl_seconds=0)")
		return nil
	}

	err := c.addEntry("", requestID, model, query, requestBody, responseBody, ttlSeconds)

	if err != nil {
		metrics.RecordCacheOperation("redis", "add_entry", "error", time.Since(start).Seconds())
	} else {
		metrics.RecordCacheOperation("redis", "add_entry", "success", time.Since(start).Seconds())
	}

	return err
}

// floatsToBytes converts float32 slice to byte array for Redis vector storage
func floatsToBytes(fs []float32) []byte {
	buf := make([]byte, len(fs)*4)
	for i, f := range fs {
		u := math.Float32bits(f)
		binary.LittleEndian.PutUint32(buf[i*4:], u)
	}
	return buf
}

// addEntry handles the internal logic for storing entries in Redis
func (c *RedisCache) addEntry(id string, requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	logging.Infof("addEntry called: id='%s', requestID='%s', requestBody_len=%d, responseBody_len=%d, ttl_seconds=%d",
		id, requestID, len(requestBody), len(responseBody), ttlSeconds)

	// Determine effective TTL: use provided value or fall back to cache default
	effectiveTTL := ttlSeconds
	if ttlSeconds == -1 {
		effectiveTTL = c.ttlSeconds
	}

	// Generate semantic embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 0)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Generate unique ID if not provided
	if id == "" {
		id = fmt.Sprintf("%x", md5.Sum(fmt.Appendf(nil, "%s_%s_%d", model, query, time.Now().UnixNano())))
	}

	ctx := context.Background()

	// Convert embedding to bytes
	embeddingBytes := floatsToBytes(embedding)

	// Prepare document key with prefix (check if already prefixed to avoid double prefix)
	var docKey string
	if strings.HasPrefix(id, c.config.Index.Prefix) {
		docKey = id // Already has prefix, use as-is
		logging.Infof("ID already has prefix, using as-is: %s", docKey)
	} else {
		docKey = c.config.Index.Prefix + id // Add prefix
		logging.Infof("Adding prefix to ID: %s -> %s", id, docKey)
	}

	responseBodyStr := string(responseBody)
	logging.Infof("Setting response_body field: len=%d, isEmpty=%v", len(responseBodyStr), responseBodyStr == "")

	// Prepare hash fields including ttl_seconds
	hashFields := map[string]interface{}{
		"request_id":                    requestID,
		"model":                         model,
		"query":                         query,
		"request_body":                  string(requestBody),
		"response_body":                 responseBodyStr,
		c.config.Index.VectorField.Name: embeddingBytes,
		"timestamp":                     time.Now().Unix(),
		"ttl_seconds":                   effectiveTTL,
	}

	// Store as Redis hash
	err = c.client.HSet(ctx, docKey, hashFields).Err()
	if err != nil {
		logging.Debugf("RedisCache.addEntry: HSet failed: %v", err)
		return fmt.Errorf("failed to store cache entry: %w", err)
	}

	// Set TTL if configured (Redis native TTL)
	if effectiveTTL > 0 {
		c.client.Expire(ctx, docKey, time.Duration(effectiveTTL)*time.Second)
	}

	logging.Debugf("RedisCache.addEntry: successfully added entry to Redis (key: %s, embedding_dim: %d, request_size: %d, response_size: %d, ttl=%d)",
		docKey, len(embedding), len(requestBody), len(responseBody), effectiveTTL)
	logging.LogEvent("cache_entry_added", map[string]interface{}{
		"backend":             "redis",
		"index":               c.indexName,
		"request_id":          requestID,
		"query":               query,
		"model":               model,
		"embedding_dimension": len(embedding),
	})
	return nil
}

// FindSimilar searches for semantically similar cached requests
func (c *RedisCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *RedisCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	start := time.Now()

	logging.Infof("FindSimilarWithThreshold ENTERED: model=%s, query='%s', threshold=%.2f", model, query, threshold)

	if !c.enabled {
		logging.Infof("FindSimilarWithThreshold: cache disabled, returning early")
		return nil, false, nil
	}

	logging.Infof("FindSimilarWithThreshold: cache enabled, generating embedding for query")

	// Generate semantic embedding for similarity comparison
	queryEmbedding, err := candle_binding.GetEmbedding(query, 0)
	if err != nil {
		metrics.RecordCacheOperation("redis", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	ctx := context.Background()

	// Convert embedding to bytes for Redis query
	embeddingBytes := floatsToBytes(queryEmbedding)

	// Build KNN query without model filter (model filtering removed for cross-model cache sharing)
	knnQuery := fmt.Sprintf("[KNN %d @%s $vec AS vector_distance]",
		c.config.Search.TopK, c.config.Index.VectorField.Name)

	// Execute vector search
	searchResult, err := c.client.FTSearchWithArgs(ctx,
		c.indexName,
		knnQuery,
		&redis.FTSearchOptions{
			Return: []redis.FTSearchReturn{
				{FieldName: "vector_distance"},
				{FieldName: "response_body"},
			},
			DialectVersion: 2,
			Params: map[string]interface{}{
				"vec": embeddingBytes,
			},
		},
	).Result()
	if err != nil {
		logging.Infof("RedisCache.FindSimilarWithThreshold: search failed: %v", err)
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("redis", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, nil
	}

	logging.Infof("RedisCache.FindSimilarWithThreshold: search returned %d results", searchResult.Total)

	if searchResult.Total == 0 {
		atomic.AddInt64(&c.missCount, 1)
		logging.Infof("RedisCache.FindSimilarWithThreshold: no entries found - cache miss")
		metrics.RecordCacheOperation("redis", "find_similar", "miss", time.Since(start).Seconds())
		return nil, false, nil
	}

	// Get best match
	bestDoc := searchResult.Docs[0]

	logging.Infof("Extracting fields from best match document...")

	// Extract distance and convert to similarity score
	// Redis returns distance, we need to convert based on metric type
	distanceVal, ok := bestDoc.Fields["vector_distance"]
	if !ok {
		logging.Infof("RedisCache.FindSimilarWithThreshold: vector_distance field not found in result")
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("redis", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, nil
	}

	var distance float64
	if _, err := fmt.Sscanf(fmt.Sprint(distanceVal), "%f", &distance); err != nil {
		logging.Infof("RedisCache.FindSimilarWithThreshold: failed to parse distance value: %v", err)
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("redis", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, nil
	}

	// Convert distance to similarity score based on metric type
	var similarity float32
	switch c.config.Index.VectorField.MetricType {
	case "COSINE":
		// COSINE distance in range [0, 2], convert to similarity [0, 1]
		similarity = 1.0 - float32(distance)/2.0
	case "IP":
		// Inner product: higher is more similar, convert appropriately
		similarity = float32(distance)
	case "L2":
		// L2 distance: lower is more similar, convert to similarity
		// Assume max distance for normalization (this is dataset dependent)
		similarity = 1.0 / (1.0 + float32(distance))
	default:
		similarity = 1.0 - float32(distance)
	}

	logging.Infof("Calculated similarity=%.4f, threshold=%.4f, distance=%.4f (metric=%s)",
		similarity, threshold, distance, c.config.Index.VectorField.MetricType)

	if similarity < threshold {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("RedisCache.FindSimilarWithThreshold: cache miss - similarity %.4f below threshold %.4f",
			similarity, threshold)
		logging.LogEvent("cache_miss", map[string]interface{}{
			"backend":         "redis",
			"best_similarity": similarity,
			"threshold":       threshold,
			"model":           model,
			"index":           c.indexName,
		})
		metrics.RecordCacheOperation("redis", "find_similar", "miss", time.Since(start).Seconds())
		return nil, false, nil
	}

	// Extract response body from cache hit
	logging.Infof("Attempting to extract response_body field...")
	responseBodyVal, ok := bestDoc.Fields["response_body"]
	if !ok {
		logging.Infof("RedisCache.FindSimilarWithThreshold: cache hit BUT response_body field is MISSING - treating as miss")
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("redis", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, nil
	}

	responseBodyStr := fmt.Sprint(responseBodyVal)
	if responseBodyStr == "" {
		logging.Infof("RedisCache.FindSimilarWithThreshold: cache hit BUT response_body is EMPTY - treating as miss")
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("redis", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, nil
	}

	logging.Infof("CACHE HIT: Found cached response, similarity=%.4f, response_size=%d bytes", similarity, len(responseBodyStr))

	responseBody := []byte(responseBodyStr)

	atomic.AddInt64(&c.hitCount, 1)
	logging.Debugf("RedisCache.FindSimilarWithThreshold: cache hit - similarity=%.4f, response_size=%d bytes",
		similarity, len(responseBody))
	logging.LogEvent("cache_hit", map[string]interface{}{
		"backend":    "redis",
		"similarity": similarity,
		"threshold":  threshold,
		"model":      model,
		"index":      c.indexName,
	})
	metrics.RecordCacheOperation("redis", "find_similar", "hit", time.Since(start).Seconds())
	return responseBody, true, nil
}

// Close releases all resources held by the cache
func (c *RedisCache) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}

// GetStats provides current cache performance metrics
func (c *RedisCache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	hits := atomic.LoadInt64(&c.hitCount)
	misses := atomic.LoadInt64(&c.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	// Retrieve index statistics from Redis
	totalEntries := 0
	if c.enabled && c.client != nil {
		ctx := context.Background()
		info, err := c.client.FTInfo(ctx, c.indexName).Result()
		if err == nil {
			// Extract document count from FTInfoResult
			totalEntries = info.NumDocs
			logging.Debugf("RedisCache.GetStats: index '%s' contains %d entries",
				c.indexName, totalEntries)
		} else {
			logging.Debugf("RedisCache.GetStats: failed to get index stats: %v", err)
		}
	}

	cacheStats := CacheStats{
		TotalEntries: totalEntries,
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}

	if c.lastCleanupTime != nil {
		cacheStats.LastCleanupTime = c.lastCleanupTime
	}

	return cacheStats
}
