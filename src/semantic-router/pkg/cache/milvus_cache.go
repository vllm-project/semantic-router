package cache

import (
	"context"
	"crypto/md5"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"sigs.k8s.io/yaml"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// MilvusConfig defines the complete configuration structure for Milvus cache backend
type MilvusConfig struct {
	Connection struct {
		Host     string `yaml:"host"`
		Port     int    `yaml:"port"`
		Database string `yaml:"database"`
		Timeout  int    `yaml:"timeout"`
		Auth     struct {
			Enabled  bool   `yaml:"enabled"`
			Username string `yaml:"username"`
			Password string `yaml:"password"`
		} `yaml:"auth"`
		TLS struct {
			Enabled  bool   `yaml:"enabled"`
			CertFile string `yaml:"cert_file"`
			KeyFile  string `yaml:"key_file"`
			CAFile   string `yaml:"ca_file"`
		} `yaml:"tls"`
	} `yaml:"connection"`
	Collection struct {
		Name        string `yaml:"name"`
		Description string `yaml:"description"`
		VectorField struct {
			Name       string `yaml:"name"`
			Dimension  int    `yaml:"dimension"`
			MetricType string `yaml:"metric_type"`
		} `yaml:"vector_field"`
		Index struct {
			Type   string `yaml:"type"`
			Params struct {
				M              int `yaml:"M"`
				EfConstruction int `yaml:"efConstruction"`
			} `yaml:"params"`
		} `yaml:"index"`
	} `yaml:"collection"`
	Search struct {
		Params struct {
			Ef int `yaml:"ef"`
		} `yaml:"params"`
		TopK             int    `yaml:"topk"`
		ConsistencyLevel string `yaml:"consistency_level"`
	} `yaml:"search"`
	Performance struct {
		ConnectionPool struct {
			MaxConnections     int `yaml:"max_connections"`
			MaxIdleConnections int `yaml:"max_idle_connections"`
			AcquireTimeout     int `yaml:"acquire_timeout"`
		} `yaml:"connection_pool"`
		Batch struct {
			InsertBatchSize int `yaml:"insert_batch_size"`
			Timeout         int `yaml:"timeout"`
		} `yaml:"batch"`
	} `yaml:"performance"`
	DataManagement struct {
		TTL struct {
			Enabled         bool   `yaml:"enabled"`
			TimestampField  string `yaml:"timestamp_field"`
			CleanupInterval int    `yaml:"cleanup_interval"`
		} `yaml:"ttl"`
		Compaction struct {
			Enabled  bool `yaml:"enabled"`
			Interval int  `yaml:"interval"`
		} `yaml:"compaction"`
	} `yaml:"data_management"`
	Logging struct {
		Level          string `yaml:"level"`
		EnableQueryLog bool   `yaml:"enable_query_log"`
		EnableMetrics  bool   `yaml:"enable_metrics"`
	} `yaml:"logging"`
	Development struct {
		DropCollectionOnStartup bool `yaml:"drop_collection_on_startup"`
		AutoCreateCollection    bool `yaml:"auto_create_collection"`
		VerboseErrors           bool `yaml:"verbose_errors"`
	} `yaml:"development"`
}

// MilvusCache provides a scalable semantic cache implementation using Milvus vector database
type MilvusCache struct {
	client              client.Client
	config              *MilvusConfig
	collectionName      string
	similarityThreshold float32
	ttlSeconds          int
	enabled             bool
	hitCount            int64
	missCount           int64
	lastCleanupTime     *time.Time
	mu                  sync.RWMutex
}

// MilvusCacheOptions contains configuration parameters for Milvus cache initialization
type MilvusCacheOptions struct {
	SimilarityThreshold float32
	TTLSeconds          int
	Enabled             bool
	ConfigPath          string
}

// NewMilvusCache initializes a new Milvus-backed semantic cache instance
func NewMilvusCache(options MilvusCacheOptions) (*MilvusCache, error) {
	if !options.Enabled {
		observability.Debugf("MilvusCache: disabled, returning stub")
		return &MilvusCache{
			enabled: false,
		}, nil
	}

	// Load Milvus configuration from file
	observability.Debugf("MilvusCache: loading config from %s", options.ConfigPath)
	config, err := loadMilvusConfig(options.ConfigPath)
	if err != nil {
		observability.Debugf("MilvusCache: failed to load config: %v", err)
		return nil, fmt.Errorf("failed to load Milvus config: %w", err)
	}
	observability.Debugf("MilvusCache: config loaded - host=%s:%d, collection=%s, dimension=auto-detect",
		config.Connection.Host, config.Connection.Port, config.Collection.Name)

	// Establish connection to Milvus server
	connectionString := fmt.Sprintf("%s:%d", config.Connection.Host, config.Connection.Port)
	observability.Debugf("MilvusCache: connecting to Milvus at %s", connectionString)
	dialCtx := context.Background()
	var cancel context.CancelFunc
	if config.Connection.Timeout > 0 {
		// If a timeout is specified, apply it to the connection context
		timeout := time.Duration(config.Connection.Timeout) * time.Second
		dialCtx, cancel = context.WithTimeout(dialCtx, timeout)
		defer cancel()
		observability.Debugf("MilvusCache: connection timeout set to %s", timeout)
	}
	milvusClient, err := client.NewGrpcClient(dialCtx, connectionString)
	if err != nil {
		observability.Debugf("MilvusCache: failed to connect: %v", err)
		return nil, fmt.Errorf("failed to create Milvus client: %w", err)
	}
	observability.Debugf("MilvusCache: successfully connected to Milvus")

	cache := &MilvusCache{
		client:              milvusClient,
		config:              config,
		collectionName:      config.Collection.Name,
		similarityThreshold: options.SimilarityThreshold,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
	}

	// Set up the collection for caching
	observability.Debugf("MilvusCache: initializing collection '%s'", config.Collection.Name)
	if err := cache.initializeCollection(); err != nil {
		observability.Debugf("MilvusCache: failed to initialize collection: %v", err)
		milvusClient.Close()
		return nil, fmt.Errorf("failed to initialize collection: %w", err)
	}
	observability.Debugf("MilvusCache: initialization complete")

	return cache, nil
}

// loadMilvusConfig reads and parses the Milvus configuration from file
func loadMilvusConfig(configPath string) (*MilvusConfig, error) {
	if configPath == "" {
		return nil, fmt.Errorf("milvus config path is required")
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config MilvusConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return &config, nil
}

// initializeCollection sets up the Milvus collection and index structures
func (c *MilvusCache) initializeCollection() error {
	ctx := context.Background()

	// Verify collection existence
	hasCollection, err := c.client.HasCollection(ctx, c.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	// Handle development mode collection reset
	if c.config.Development.DropCollectionOnStartup && hasCollection {
		if err := c.client.DropCollection(ctx, c.collectionName); err != nil {
			observability.Debugf("MilvusCache: failed to drop collection: %v", err)
			return fmt.Errorf("failed to drop collection: %w", err)
		}
		hasCollection = false
		observability.Debugf("MilvusCache: dropped existing collection '%s' for development", c.collectionName)
		observability.LogEvent("collection_dropped", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"reason":     "development_mode",
		})
	}

	// Create collection if it doesn't exist
	if !hasCollection {
		if !c.config.Development.AutoCreateCollection {
			return fmt.Errorf("collection %s does not exist and auto-creation is disabled", c.collectionName)
		}

		if err := c.createCollection(); err != nil {
			observability.Debugf("MilvusCache: failed to create collection: %v", err)
			return fmt.Errorf("failed to create collection: %w", err)
		}
		observability.Debugf("MilvusCache: created new collection '%s' with dimension %d",
			c.collectionName, c.config.Collection.VectorField.Dimension)
		observability.LogEvent("collection_created", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"dimension":  c.config.Collection.VectorField.Dimension,
		})
	}

	// Load collection into memory for queries
	observability.Debugf("MilvusCache: loading collection '%s' into memory", c.collectionName)
	if err := c.client.LoadCollection(ctx, c.collectionName, false); err != nil {
		observability.Debugf("MilvusCache: failed to load collection: %v", err)
		return fmt.Errorf("failed to load collection: %w", err)
	}
	observability.Debugf("MilvusCache: collection loaded successfully")

	return nil
}

// createCollection builds the Milvus collection with the appropriate schema
func (c *MilvusCache) createCollection() error {
	ctx := context.Background()

	// Determine embedding dimension automatically
	testEmbedding, err := candle_binding.GetEmbedding("test", 0) // Auto-detect
	if err != nil {
		return fmt.Errorf("failed to detect embedding dimension: %w", err)
	}
	actualDimension := len(testEmbedding)

	observability.Debugf("MilvusCache.createCollection: auto-detected embedding dimension: %d", actualDimension)

	// Define schema with auto-detected dimension
	schema := &entity.Schema{
		CollectionName: c.collectionName,
		Description:    c.config.Collection.Description,
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				TypeParams: map[string]string{"max_length": "64"},
			},
			{
				Name:       "request_id",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "64"},
			},
			{
				Name:       "model",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "256"},
			},
			{
				Name:       "query",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:       "request_body",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:       "response_body",
				DataType:   entity.FieldTypeVarChar,
				TypeParams: map[string]string{"max_length": "65535"},
			},
			{
				Name:     c.config.Collection.VectorField.Name,
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", actualDimension), // Use auto-detected dimension
				},
			},
			{
				Name:     "timestamp",
				DataType: entity.FieldTypeInt64,
			},
		},
	}

	// Create collection
	if createErr := c.client.CreateCollection(ctx, schema, 1); createErr != nil {
		return createErr
	}

	// Create index with updated API
	index, err := entity.NewIndexHNSW(entity.MetricType(c.config.Collection.VectorField.MetricType), c.config.Collection.Index.Params.EfConstruction, c.config.Collection.Index.Params.M)
	if err != nil {
		return fmt.Errorf("failed to create HNSW index: %w", err)
	}
	if err := c.client.CreateIndex(ctx, c.collectionName, c.config.Collection.VectorField.Name, index, false); err != nil {
		return err
	}

	return nil
}

// IsEnabled returns the current cache activation status
func (c *MilvusCache) IsEnabled() bool {
	return c.enabled
}

// AddPendingRequest stores a request that is awaiting its response
func (c *MilvusCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Store incomplete entry for later completion with response
	err := c.addEntry("", requestID, model, query, requestBody, nil)

	if err != nil {
		metrics.RecordCacheOperation("milvus", "add_pending", "error", time.Since(start).Seconds())
	} else {
		metrics.RecordCacheOperation("milvus", "add_pending", "success", time.Since(start).Seconds())
	}

	return err
}

// UpdateWithResponse completes a pending request by adding the response
func (c *MilvusCache) UpdateWithResponse(requestID string, responseBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	observability.Debugf("MilvusCache.UpdateWithResponse: updating pending entry (request_id: %s, response_size: %d)",
		requestID, len(responseBody))

	// Find the pending entry and complete it with the response
	// Query for the incomplete entry to retrieve its metadata
	ctx := context.Background()
	queryExpr := fmt.Sprintf("request_id == \"%s\" && response_body == \"\"", requestID)

	observability.Debugf("MilvusCache.UpdateWithResponse: searching for pending entry with expr: %s", queryExpr)

	results, err := c.client.Query(ctx, c.collectionName, []string{}, queryExpr,
		[]string{"id", "model", "query", "request_body"})
	if err != nil {
		observability.Debugf("MilvusCache.UpdateWithResponse: query failed: %v", err)
		metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to query pending entry: %w", err)
	}

	if len(results) == 0 {
		observability.Debugf("MilvusCache.UpdateWithResponse: no pending entry found")
		metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("no pending entry found")
	}

	// Get the model and request body from the pending entry
	idColumn := results[0].(*entity.ColumnVarChar)
	modelColumn := results[1].(*entity.ColumnVarChar)
	queryColumn := results[2].(*entity.ColumnVarChar)
	requestColumn := results[3].(*entity.ColumnVarChar)

	if idColumn.Len() > 0 {
		id := idColumn.Data()[0]
		model := modelColumn.Data()[0]
		query := queryColumn.Data()[0]
		requestBody := requestColumn.Data()[0]

		observability.Debugf("MilvusCache.UpdateWithResponse: found pending entry, adding complete entry (id: %s, model: %s)", id, model)

		// Create the complete entry with response data
		err := c.addEntry(id, requestID, model, query, []byte(requestBody), responseBody)
		if err != nil {
			metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
			return fmt.Errorf("failed to add complete entry: %w", err)
		}

		observability.Debugf("MilvusCache.UpdateWithResponse: successfully added complete entry with response")
		metrics.RecordCacheOperation("milvus", "update_response", "success", time.Since(start).Seconds())
	}

	return nil
}

// AddEntry stores a complete request-response pair in the cache
func (c *MilvusCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	err := c.addEntry("", requestID, model, query, requestBody, responseBody)

	if err != nil {
		metrics.RecordCacheOperation("milvus", "add_entry", "error", time.Since(start).Seconds())
	} else {
		metrics.RecordCacheOperation("milvus", "add_entry", "success", time.Since(start).Seconds())
	}

	return err
}

// addEntry handles the internal logic for storing entries in Milvus
func (c *MilvusCache) addEntry(id string, requestID string, model string, query string, requestBody, responseBody []byte) error {
	// Generate semantic embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Generate unique ID if not provided
	if id == "" {
		id = fmt.Sprintf("%x", md5.Sum(fmt.Appendf(nil, "%s_%s_%d", model, query, time.Now().UnixNano())))
	}

	ctx := context.Background()

	// Prepare data for upsert
	ids := []string{id}
	requestIDs := []string{requestID}
	models := []string{model}
	queries := []string{query}
	requestBodies := []string{string(requestBody)}
	responseBodies := []string{string(responseBody)}
	embeddings := [][]float32{embedding}
	timestamps := []int64{time.Now().Unix()}

	// Create columns
	idColumn := entity.NewColumnVarChar("id", ids)
	requestIDColumn := entity.NewColumnVarChar("request_id", requestIDs)
	modelColumn := entity.NewColumnVarChar("model", models)
	queryColumn := entity.NewColumnVarChar("query", queries)
	requestColumn := entity.NewColumnVarChar("request_body", requestBodies)
	responseColumn := entity.NewColumnVarChar("response_body", responseBodies)
	embeddingColumn := entity.NewColumnFloatVector(c.config.Collection.VectorField.Name, len(embedding), embeddings)
	timestampColumn := entity.NewColumnInt64("timestamp", timestamps)

	// Upsert the entry into the collection
	observability.Debugf("MilvusCache.addEntry: upserting entry into collection '%s' (embedding_dim: %d, request_size: %d, response_size: %d)",
		c.collectionName, len(embedding), len(requestBody), len(responseBody))
	_, err = c.client.Upsert(ctx, c.collectionName, "", idColumn, requestIDColumn, modelColumn, queryColumn, requestColumn, responseColumn, embeddingColumn, timestampColumn)
	if err != nil {
		observability.Debugf("MilvusCache.addEntry: upsert failed: %v", err)
		return fmt.Errorf("failed to upsert cache entry: %w", err)
	}

	// Ensure data is persisted to storage
	if err := c.client.Flush(ctx, c.collectionName, false); err != nil {
		observability.Warnf("Failed to flush cache entry: %v", err)
	}

	observability.Debugf("MilvusCache.addEntry: successfully added entry to Milvus")
	observability.LogEvent("cache_entry_added", map[string]interface{}{
		"backend":             "milvus",
		"collection":          c.collectionName,
		"request_id":          requestID,
		"query":               query,
		"model":               model,
		"embedding_dimension": len(embedding),
	})
	return nil
}

// FindSimilar searches for semantically similar cached requests
func (c *MilvusCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *MilvusCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	start := time.Now()

	if !c.enabled {
		observability.Debugf("MilvusCache.FindSimilarWithThreshold: cache disabled")
		return nil, false, nil
	}
	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	observability.Debugf("MilvusCache.FindSimilarWithThreshold: searching for model='%s', query='%s' (len=%d chars), threshold=%.4f",
		model, queryPreview, len(query), threshold)

	// Generate semantic embedding for similarity comparison
	queryEmbedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	ctx := context.Background()

	// Define search parameters
	searchParam, err := entity.NewIndexHNSWSearchParam(c.config.Search.Params.Ef)
	if err != nil {
		return nil, false, fmt.Errorf("failed to create search parameters: %w", err)
	}

	// Use Milvus Search for efficient similarity search
	searchResult, err := c.client.Search(
		ctx,
		c.collectionName,
		[]string{},
		fmt.Sprintf("model == \"%s\" && response_body != \"\"", model),
		[]string{"response_body"},
		[]entity.Vector{entity.FloatVector(queryEmbedding)},
		c.config.Collection.VectorField.Name,
		entity.MetricType(c.config.Collection.VectorField.MetricType),
		c.config.Search.TopK,
		searchParam,
	)
	if err != nil {
		observability.Debugf("MilvusCache.FindSimilarWithThreshold: search failed: %v", err)
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		atomic.AddInt64(&c.missCount, 1)
		observability.Debugf("MilvusCache.FindSimilarWithThreshold: no entries found")
		metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	bestScore := searchResult[0].Scores[0]
	if bestScore < threshold {
		atomic.AddInt64(&c.missCount, 1)
		observability.Debugf("MilvusCache.FindSimilarWithThreshold: CACHE MISS - best_similarity=%.4f < threshold=%.4f",
			bestScore, threshold)
		observability.LogEvent("cache_miss", map[string]interface{}{
			"backend":         "milvus",
			"best_similarity": bestScore,
			"threshold":       threshold,
			"model":           model,
			"collection":      c.collectionName,
		})
		metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	// Cache Hit
	var responseBody []byte
	responseBodyColumn, ok := searchResult[0].Fields[0].(*entity.ColumnVarChar)
	if ok && responseBodyColumn.Len() > 0 {
		responseBody = []byte(responseBodyColumn.Data()[0])
	}

	if responseBody == nil {
		observability.Debugf("MilvusCache.FindSimilarWithThreshold: cache hit but response_body is missing or not a string")
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	atomic.AddInt64(&c.hitCount, 1)
	observability.Debugf("MilvusCache.FindSimilarWithThreshold: CACHE HIT - similarity=%.4f >= threshold=%.4f, response_size=%d bytes",
		bestScore, threshold, len(responseBody))
	observability.LogEvent("cache_hit", map[string]interface{}{
		"backend":    "milvus",
		"similarity": bestScore,
		"threshold":  threshold,
		"model":      model,
		"collection": c.collectionName,
	})
	metrics.RecordCacheOperation("milvus", "find_similar", "hit", time.Since(start).Seconds())
	metrics.RecordCacheHit()
	return responseBody, true, nil
}

// Close releases all resources held by the cache
func (c *MilvusCache) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}

// GetStats provides current cache performance metrics
func (c *MilvusCache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	hits := atomic.LoadInt64(&c.hitCount)
	misses := atomic.LoadInt64(&c.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	// Retrieve collection statistics from Milvus
	totalEntries := 0
	if c.enabled && c.client != nil {
		ctx := context.Background()
		stats, err := c.client.GetCollectionStatistics(ctx, c.collectionName)
		if err == nil {
			// Extract entity count from statistics
			if entityCount, ok := stats["row_count"]; ok {
				_, _ = fmt.Sscanf(entityCount, "%d", &totalEntries)
				observability.Debugf("MilvusCache.GetStats: collection '%s' contains %d entries",
					c.collectionName, totalEntries)
			}
		} else {
			observability.Debugf("MilvusCache.GetStats: failed to get collection stats: %v", err)
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
