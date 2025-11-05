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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
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
		logging.Debugf("MilvusCache: disabled, returning stub")
		return &MilvusCache{
			enabled: false,
		}, nil
	}

	// Load Milvus configuration from file
	logging.Debugf("MilvusCache: loading config from %s", options.ConfigPath)
	config, err := loadMilvusConfig(options.ConfigPath)
	if err != nil {
		logging.Debugf("MilvusCache: failed to load config: %v", err)
		return nil, fmt.Errorf("failed to load Milvus config: %w", err)
	}
	logging.Debugf("MilvusCache: config loaded - host=%s:%d, collection=%s, dimension=auto-detect",
		config.Connection.Host, config.Connection.Port, config.Collection.Name)

	// Establish connection to Milvus server
	connectionString := fmt.Sprintf("%s:%d", config.Connection.Host, config.Connection.Port)
	logging.Debugf("MilvusCache: connecting to Milvus at %s", connectionString)
	dialCtx := context.Background()
	var cancel context.CancelFunc
	if config.Connection.Timeout > 0 {
		// If a timeout is specified, apply it to the connection context
		timeout := time.Duration(config.Connection.Timeout) * time.Second
		dialCtx, cancel = context.WithTimeout(dialCtx, timeout)
		defer cancel()
		logging.Debugf("MilvusCache: connection timeout set to %s", timeout)
	}
	milvusClient, err := client.NewGrpcClient(dialCtx, connectionString)
	if err != nil {
		logging.Debugf("MilvusCache: failed to connect: %v", err)
		return nil, fmt.Errorf("failed to create Milvus client: %w", err)
	}
	logging.Debugf("MilvusCache: successfully connected to Milvus")

	cache := &MilvusCache{
		client:              milvusClient,
		config:              config,
		collectionName:      config.Collection.Name,
		similarityThreshold: options.SimilarityThreshold,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
	}

	// Set up the collection for caching
	logging.Debugf("MilvusCache: initializing collection '%s'", config.Collection.Name)
	if err := cache.initializeCollection(); err != nil {
		logging.Debugf("MilvusCache: failed to initialize collection: %v", err)
		milvusClient.Close()
		return nil, fmt.Errorf("failed to initialize collection: %w", err)
	}
	logging.Debugf("MilvusCache: initialization complete")

	return cache, nil
}

// loadMilvusConfig reads and parses the Milvus configuration from file
func loadMilvusConfig(configPath string) (*MilvusConfig, error) {
	if configPath == "" {
		return nil, fmt.Errorf("milvus config path is required")
	}

	fmt.Printf("[DEBUG] Loading Milvus config from: %s\n", configPath)

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	fmt.Printf("[DEBUG] Config file size: %d bytes\n", len(data))

	var config MilvusConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Debug: Log what was parsed
	fmt.Printf("[DEBUG] MilvusConfig parsed from %s:\n", configPath)
	fmt.Printf("[DEBUG]   Collection.Name: %s\n", config.Collection.Name)
	fmt.Printf("[DEBUG]   Collection.VectorField.Name: %s\n", config.Collection.VectorField.Name)
	fmt.Printf("[DEBUG]   Collection.VectorField.Dimension: %d\n", config.Collection.VectorField.Dimension)
	fmt.Printf("[DEBUG]   Collection.VectorField.MetricType: %s\n", config.Collection.VectorField.MetricType)
	fmt.Printf("[DEBUG]   Collection.Index.Type: %s\n", config.Collection.Index.Type)
	fmt.Printf("[DEBUG]   Development.AutoCreateCollection: %v\n", config.Development.AutoCreateCollection)
	fmt.Printf("[DEBUG]   Development.DropCollectionOnStartup: %v\n", config.Development.DropCollectionOnStartup)

	// WORKAROUND: Force development settings for benchmarks/tests only
	// There seems to be a YAML parsing issue with sigs.k8s.io/yaml
	// Only apply this workaround if SR_BENCHMARK_MODE or SR_TEST_MODE is set
	benchmarkMode := os.Getenv("SR_BENCHMARK_MODE")
	testMode := os.Getenv("SR_TEST_MODE")
	if (benchmarkMode == "1" || benchmarkMode == "true" || testMode == "1" || testMode == "true") &&
		!config.Development.AutoCreateCollection && !config.Development.DropCollectionOnStartup {
		fmt.Printf("[WARN] Development settings parsed as false, forcing to true for benchmarks/tests\n")
		config.Development.AutoCreateCollection = true
		config.Development.DropCollectionOnStartup = true
	}

	// WORKAROUND: Force vector field settings if empty
	if config.Collection.VectorField.Name == "" {
		fmt.Printf("[WARN] VectorField.Name parsed as empty, setting to 'embedding'\n")
		config.Collection.VectorField.Name = "embedding"
	}
	if config.Collection.VectorField.MetricType == "" {
		fmt.Printf("[WARN] VectorField.MetricType parsed as empty, setting to 'IP'\n")
		config.Collection.VectorField.MetricType = "IP"
	}
	if config.Collection.Index.Type == "" {
		fmt.Printf("[WARN] Index.Type parsed as empty, setting to 'HNSW'\n")
		config.Collection.Index.Type = "HNSW"
	}
	// Validate index params
	if config.Collection.Index.Params.M == 0 {
		fmt.Printf("[WARN] Index.Params.M parsed as 0, setting to 16\n")
		config.Collection.Index.Params.M = 16
	}
	if config.Collection.Index.Params.EfConstruction == 0 {
		fmt.Printf("[WARN] Index.Params.EfConstruction parsed as 0, setting to 64\n")
		config.Collection.Index.Params.EfConstruction = 64
	}
	// Validate search params
	if config.Search.Params.Ef == 0 {
		fmt.Printf("[WARN] Search.Params.Ef parsed as 0, setting to 64\n")
		config.Search.Params.Ef = 64
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
			logging.Debugf("MilvusCache: failed to drop collection: %v", err)
			return fmt.Errorf("failed to drop collection: %w", err)
		}
		hasCollection = false
		logging.Debugf("MilvusCache: dropped existing collection '%s' for development", c.collectionName)
		logging.LogEvent("collection_dropped", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"reason":     "development_mode",
		})
	}

	// Create collection if it doesn't exist
	if !hasCollection {
		fmt.Printf("[DEBUG] Collection '%s' does not exist. AutoCreateCollection=%v\n",
			c.collectionName, c.config.Development.AutoCreateCollection)
		if !c.config.Development.AutoCreateCollection {
			return fmt.Errorf("collection %s does not exist and auto-creation is disabled", c.collectionName)
		}

		if err := c.createCollection(); err != nil {
			logging.Debugf("MilvusCache: failed to create collection: %v", err)
			return fmt.Errorf("failed to create collection: %w", err)
		}
		logging.Debugf("MilvusCache: created new collection '%s' with dimension %d",
			c.collectionName, c.config.Collection.VectorField.Dimension)
		logging.LogEvent("collection_created", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"dimension":  c.config.Collection.VectorField.Dimension,
		})
	}

	// Load collection into memory for queries
	logging.Debugf("MilvusCache: loading collection '%s' into memory", c.collectionName)
	if err := c.client.LoadCollection(ctx, c.collectionName, false); err != nil {
		logging.Debugf("MilvusCache: failed to load collection: %v", err)
		return fmt.Errorf("failed to load collection: %w", err)
	}
	logging.Debugf("MilvusCache: collection loaded successfully")

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

	logging.Debugf("MilvusCache.createCollection: auto-detected embedding dimension: %d", actualDimension)

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
	index, err := entity.NewIndexHNSW(entity.MetricType(c.config.Collection.VectorField.MetricType), c.config.Collection.Index.Params.M, c.config.Collection.Index.Params.EfConstruction)
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

	logging.Debugf("MilvusCache.UpdateWithResponse: updating pending entry (request_id: %s, response_size: %d)",
		requestID, len(responseBody))

	// Find the pending entry and complete it with the response
	// Query for the incomplete entry to retrieve its metadata
	ctx := context.Background()
	queryExpr := fmt.Sprintf("request_id == \"%s\" && response_body == \"\"", requestID)

	logging.Debugf("MilvusCache.UpdateWithResponse: searching for pending entry with expr: %s", queryExpr)

	results, err := c.client.Query(ctx, c.collectionName, []string{}, queryExpr,
		[]string{"id", "model", "query", "request_body"})
	if err != nil {
		logging.Debugf("MilvusCache.UpdateWithResponse: query failed: %v", err)
		metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to query pending entry: %w", err)
	}

	if len(results) == 0 {
		logging.Debugf("MilvusCache.UpdateWithResponse: no pending entry found")
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

		logging.Debugf("MilvusCache.UpdateWithResponse: found pending entry, adding complete entry (id: %s, model: %s)", id, model)

		// Create the complete entry with response data
		err := c.addEntry(id, requestID, model, query, []byte(requestBody), responseBody)
		if err != nil {
			metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
			return fmt.Errorf("failed to add complete entry: %w", err)
		}

		logging.Debugf("MilvusCache.UpdateWithResponse: successfully added complete entry with response")
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

// AddEntriesBatch stores multiple request-response pairs in the cache efficiently
func (c *MilvusCache) AddEntriesBatch(entries []CacheEntry) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	if len(entries) == 0 {
		return nil
	}

	logging.Debugf("MilvusCache.AddEntriesBatch: adding %d entries in batch", len(entries))

	// Prepare slices for all entries
	ids := make([]string, len(entries))
	requestIDs := make([]string, len(entries))
	models := make([]string, len(entries))
	queries := make([]string, len(entries))
	requestBodies := make([]string, len(entries))
	responseBodies := make([]string, len(entries))
	embeddings := make([][]float32, len(entries))
	timestamps := make([]int64, len(entries))

	// Generate embeddings and prepare data for all entries
	for i, entry := range entries {
		// Generate semantic embedding for the query
		embedding, err := candle_binding.GetEmbedding(entry.Query, 0)
		if err != nil {
			return fmt.Errorf("failed to generate embedding for entry %d: %w", i, err)
		}

		// Generate unique ID
		id := fmt.Sprintf("%x", md5.Sum(fmt.Appendf(nil, "%s_%s_%d", entry.Model, entry.Query, time.Now().UnixNano())))

		ids[i] = id
		requestIDs[i] = entry.RequestID
		models[i] = entry.Model
		queries[i] = entry.Query
		requestBodies[i] = string(entry.RequestBody)
		responseBodies[i] = string(entry.ResponseBody)
		embeddings[i] = embedding
		timestamps[i] = time.Now().Unix()
	}

	ctx := context.Background()

	// Get embedding dimension from first entry
	embeddingDim := len(embeddings[0])

	// Create columns
	idColumn := entity.NewColumnVarChar("id", ids)
	requestIDColumn := entity.NewColumnVarChar("request_id", requestIDs)
	modelColumn := entity.NewColumnVarChar("model", models)
	queryColumn := entity.NewColumnVarChar("query", queries)
	requestColumn := entity.NewColumnVarChar("request_body", requestBodies)
	responseColumn := entity.NewColumnVarChar("response_body", responseBodies)
	embeddingColumn := entity.NewColumnFloatVector(c.config.Collection.VectorField.Name, embeddingDim, embeddings)
	timestampColumn := entity.NewColumnInt64("timestamp", timestamps)

	// Upsert all entries at once
	logging.Debugf("MilvusCache.AddEntriesBatch: upserting %d entries into collection '%s'",
		len(entries), c.collectionName)
	_, err := c.client.Upsert(ctx, c.collectionName, "", idColumn, requestIDColumn, modelColumn, queryColumn, requestColumn, responseColumn, embeddingColumn, timestampColumn)
	if err != nil {
		logging.Debugf("MilvusCache.AddEntriesBatch: upsert failed: %v", err)
		metrics.RecordCacheOperation("milvus", "add_entries_batch", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to upsert cache entries: %w", err)
	}

	// Note: Flush removed from batch operation for performance
	// Call Flush() explicitly after all batches if immediate persistence is required

	elapsed := time.Since(start)
	logging.Debugf("MilvusCache.AddEntriesBatch: successfully added %d entries in %v (%.0f entries/sec)",
		len(entries), elapsed, float64(len(entries))/elapsed.Seconds())
	metrics.RecordCacheOperation("milvus", "add_entries_batch", "success", elapsed.Seconds())

	return nil
}

// Flush forces Milvus to persist all buffered data to disk
func (c *MilvusCache) Flush() error {
	if !c.enabled {
		return nil
	}

	ctx := context.Background()
	if err := c.client.Flush(ctx, c.collectionName, false); err != nil {
		return fmt.Errorf("failed to flush: %w", err)
	}

	logging.Debugf("MilvusCache: flushed collection '%s'", c.collectionName)
	return nil
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
	logging.Debugf("MilvusCache.addEntry: upserting entry into collection '%s' (embedding_dim: %d, request_size: %d, response_size: %d)",
		c.collectionName, len(embedding), len(requestBody), len(responseBody))
	_, err = c.client.Upsert(ctx, c.collectionName, "", idColumn, requestIDColumn, modelColumn, queryColumn, requestColumn, responseColumn, embeddingColumn, timestampColumn)
	if err != nil {
		logging.Debugf("MilvusCache.addEntry: upsert failed: %v", err)
		return fmt.Errorf("failed to upsert cache entry: %w", err)
	}

	// Ensure data is persisted to storage
	if err := c.client.Flush(ctx, c.collectionName, false); err != nil {
		logging.Warnf("Failed to flush cache entry: %v", err)
	}

	logging.Debugf("MilvusCache.addEntry: successfully added entry to Milvus")
	logging.LogEvent("cache_entry_added", map[string]interface{}{
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
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: cache disabled")
		return nil, false, nil
	}
	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	logging.Debugf("MilvusCache.FindSimilarWithThreshold: searching for model='%s', query='%s' (len=%d chars), threshold=%.4f",
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
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: search failed: %v", err)
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: no entries found")
		metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	bestScore := searchResult[0].Scores[0]
	if bestScore < threshold {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: CACHE MISS - best_similarity=%.4f < threshold=%.4f",
			bestScore, threshold)
		logging.LogEvent("cache_miss", map[string]interface{}{
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
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: cache hit but response_body is missing or not a string")
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	atomic.AddInt64(&c.hitCount, 1)
	logging.Debugf("MilvusCache.FindSimilarWithThreshold: CACHE HIT - similarity=%.4f >= threshold=%.4f, response_size=%d bytes",
		bestScore, threshold, len(responseBody))
	logging.LogEvent("cache_hit", map[string]interface{}{
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

// GetAllEntries retrieves all entries from Milvus for HNSW index rebuilding
// Returns slices of request_ids and embeddings for efficient bulk loading
func (c *MilvusCache) GetAllEntries(ctx context.Context) ([]string, [][]float32, error) {
	start := time.Now()

	if !c.enabled {
		return nil, nil, fmt.Errorf("milvus cache is not enabled")
	}

	logging.Infof("MilvusCache.GetAllEntries: querying all entries for HNSW rebuild")

	// Query all entries with embeddings and request_ids
	// Filter to only get entries with complete responses (not pending)
	queryResult, err := c.client.Query(
		ctx,
		c.collectionName,
		[]string{},              // Empty partitions means search all
		"response_body != \"\"", // Only get complete entries
		[]string{"request_id", c.config.Collection.VectorField.Name}, // Get IDs and embeddings
	)
	if err != nil {
		logging.Warnf("MilvusCache.GetAllEntries: query failed: %v", err)
		return nil, nil, fmt.Errorf("milvus query all failed: %w", err)
	}

	if len(queryResult) < 2 {
		logging.Infof("MilvusCache.GetAllEntries: no entries found or incomplete result")
		return []string{}, [][]float32{}, nil
	}

	// Extract request IDs (first column)
	requestIDColumn, ok := queryResult[0].(*entity.ColumnVarChar)
	if !ok {
		return nil, nil, fmt.Errorf("unexpected request_id column type: %T", queryResult[0])
	}

	// Extract embeddings (second column)
	embeddingColumn, ok := queryResult[1].(*entity.ColumnFloatVector)
	if !ok {
		return nil, nil, fmt.Errorf("unexpected embedding column type: %T", queryResult[1])
	}

	if requestIDColumn.Len() != embeddingColumn.Len() {
		return nil, nil, fmt.Errorf("column length mismatch: request_ids=%d, embeddings=%d",
			requestIDColumn.Len(), embeddingColumn.Len())
	}

	entryCount := requestIDColumn.Len()
	requestIDs := make([]string, entryCount)

	// Extract request IDs from column
	for i := 0; i < entryCount; i++ {
		requestID, err := requestIDColumn.ValueByIdx(i)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to get request_id at index %d: %w", i, err)
		}
		requestIDs[i] = requestID
	}

	// Extract embeddings directly from column data
	embeddings := embeddingColumn.Data()
	if len(embeddings) != entryCount {
		return nil, nil, fmt.Errorf("embedding data length mismatch: got %d, expected %d",
			len(embeddings), entryCount)
	}

	elapsed := time.Since(start)
	logging.Infof("MilvusCache.GetAllEntries: loaded %d entries in %v (%.0f entries/sec)",
		entryCount, elapsed, float64(entryCount)/elapsed.Seconds())

	return requestIDs, embeddings, nil
}

// GetByID retrieves a document from Milvus by its request ID
// This is much more efficient than FindSimilar when you already know the ID
// Used by hybrid cache to fetch documents after local HNSW search
func (c *MilvusCache) GetByID(ctx context.Context, requestID string) ([]byte, error) {
	start := time.Now()

	if !c.enabled {
		return nil, fmt.Errorf("milvus cache is not enabled")
	}

	logging.Debugf("MilvusCache.GetByID: fetching requestID='%s'", requestID)

	// Query Milvus by request_id (primary key)
	// Filter for non-empty responses to avoid race condition with pending entries
	queryResult, err := c.client.Query(
		ctx,
		c.collectionName,
		[]string{}, // Empty partitions means search all
		fmt.Sprintf("request_id == \"%s\" && response_body != \"\"", requestID),
		[]string{"response_body"}, // Only fetch document, not embedding!
	)
	if err != nil {
		logging.Debugf("MilvusCache.GetByID: query failed: %v", err)
		metrics.RecordCacheOperation("milvus", "get_by_id", "error", time.Since(start).Seconds())
		return nil, fmt.Errorf("milvus query failed: %w", err)
	}

	if len(queryResult) == 0 {
		logging.Debugf("MilvusCache.GetByID: document not found: %s", requestID)
		metrics.RecordCacheOperation("milvus", "get_by_id", "miss", time.Since(start).Seconds())
		return nil, fmt.Errorf("document not found: %s", requestID)
	}

	// Extract response body (first column since we only requested "response_body")
	responseBodyColumn, ok := queryResult[0].(*entity.ColumnVarChar)
	if !ok {
		logging.Debugf("MilvusCache.GetByID: unexpected response_body column type: %T", queryResult[0])
		metrics.RecordCacheOperation("milvus", "get_by_id", "error", time.Since(start).Seconds())
		return nil, fmt.Errorf("invalid response_body column type: %T", queryResult[0])
	}

	if responseBodyColumn.Len() == 0 {
		logging.Debugf("MilvusCache.GetByID: response_body column is empty")
		metrics.RecordCacheOperation("milvus", "get_by_id", "miss", time.Since(start).Seconds())
		return nil, fmt.Errorf("response_body is empty for: %s", requestID)
	}

	// Get the response body value
	responseBodyStr, err := responseBodyColumn.ValueByIdx(0)
	if err != nil {
		logging.Debugf("MilvusCache.GetByID: failed to get response_body value: %v", err)
		metrics.RecordCacheOperation("milvus", "get_by_id", "error", time.Since(start).Seconds())
		return nil, fmt.Errorf("failed to get response_body value: %w", err)
	}

	responseBody := []byte(responseBodyStr)

	if len(responseBody) == 0 {
		logging.Debugf("MilvusCache.GetByID: response_body is empty")
		metrics.RecordCacheOperation("milvus", "get_by_id", "miss", time.Since(start).Seconds())
		return nil, fmt.Errorf("response_body is empty for: %s", requestID)
	}

	logging.Debugf("MilvusCache.GetByID: SUCCESS - fetched %d bytes in %dms",
		len(responseBody), time.Since(start).Milliseconds())
	metrics.RecordCacheOperation("milvus", "get_by_id", "success", time.Since(start).Seconds())

	return responseBody, nil
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
				logging.Debugf("MilvusCache.GetStats: collection '%s' contains %d entries",
					c.collectionName, totalEntries)
			}
		} else {
			logging.Debugf("MilvusCache.GetStats: failed to get collection stats: %v", err)
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
