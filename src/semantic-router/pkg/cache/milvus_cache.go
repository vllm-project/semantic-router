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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// MilvusCache provides a scalable semantic cache implementation using Milvus vector database
type MilvusCache struct {
	client              client.Client
	config              *config.MilvusConfig
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
	Config              *config.MilvusConfig
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

	// (Fallback) Load Milvus configuration from a separated configuration file
	var err error
	var milvusConfig *config.MilvusConfig
	if options.Config == nil {
		logging.Warnf("(Deprecated) MilvusCache: loading config from %s", options.ConfigPath)
		milvusConfig, err = loadMilvusConfig(options.ConfigPath)
		if err != nil {
			logging.Debugf("MilvusCache: failed to load config: %v", err)
			return nil, fmt.Errorf("failed to load Milvus config: %w", err)
		}
	} else {
		milvusConfig = options.Config
	}
	logging.Debugf("MilvusCache: config loaded - host=%s:%d, collection=%s, dimension=auto-detect",
		milvusConfig.Connection.Host, milvusConfig.Connection.Port, milvusConfig.Collection.Name)

	// Establish connection to Milvus server
	connectionString := fmt.Sprintf("%s:%d", milvusConfig.Connection.Host, milvusConfig.Connection.Port)
	logging.Debugf("MilvusCache: connecting to Milvus at %s", connectionString)
	dialCtx := context.Background()
	var cancel context.CancelFunc
	if milvusConfig.Connection.Timeout > 0 {
		// If a timeout is specified, apply it to the connection context
		timeout := time.Duration(milvusConfig.Connection.Timeout) * time.Second
		dialCtx, cancel = context.WithTimeout(dialCtx, timeout)
		defer cancel()
		logging.Debugf("MilvusCache: connection timeout set to %s", timeout)
	}
	milvusClient, err := client.NewGrpcClient(dialCtx, connectionString)
	if err != nil {
		logging.Debugf("MilvusCache: failed to connect: %v", err)
		return nil, fmt.Errorf("failed to create Milvus client: %w", err)
	}

	cache := &MilvusCache{
		client:              milvusClient,
		config:              milvusConfig,
		collectionName:      milvusConfig.Collection.Name,
		similarityThreshold: options.SimilarityThreshold,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
	}

	// Test connection using the new CheckConnection method
	if err := cache.CheckConnection(); err != nil {
		logging.Debugf("MilvusCache: connection check failed: %v", err)
		milvusClient.Close()
		return nil, err
	}
	logging.Debugf("MilvusCache: successfully connected to Milvus")

	// Set up the collection for caching
	logging.Debugf("MilvusCache: initializing collection '%s'", milvusConfig.Collection.Name)
	if err := cache.initializeCollection(); err != nil {
		logging.Debugf("MilvusCache: failed to initialize collection: %v", err)
		milvusClient.Close()
		return nil, fmt.Errorf("failed to initialize collection: %w", err)
	}
	logging.Debugf("MilvusCache: initialization complete")

	return cache, nil
}

// loadMilvusConfig reads and parses the Milvus configuration from file (Deprecated)
func loadMilvusConfig(configPath string) (*config.MilvusConfig, error) {
	if configPath == "" {
		return nil, fmt.Errorf("milvus config path is required")
	}

	logging.Debugf("Loading Milvus config from: %s\n", configPath)

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	logging.Debugf("Config file size: %d bytes\n", len(data))

	var milvusConfig *config.MilvusConfig
	if err = yaml.Unmarshal(data, &milvusConfig); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Debug: Log what was parsed
	logging.Debugf("MilvusConfig parsed from %s:\n", configPath)
	logging.Debugf("Collection.Name: %s\n", milvusConfig.Collection.Name)
	logging.Debugf("Collection.VectorField.Name: %s\n", milvusConfig.Collection.VectorField.Name)
	logging.Debugf("Collection.VectorField.Dimension: %d\n", milvusConfig.Collection.VectorField.Dimension)
	logging.Debugf("Collection.VectorField.MetricType: %s\n", milvusConfig.Collection.VectorField.MetricType)
	logging.Debugf("Collection.Index.Type: %s\n", milvusConfig.Collection.Index.Type)
	logging.Debugf("Development.AutoCreateCollection: %v\n", milvusConfig.Development.AutoCreateCollection)
	logging.Debugf("Development.DropCollectionOnStartup: %v\n", milvusConfig.Development.DropCollectionOnStartup)

	// WORKAROUND: Force development settings for benchmarks/tests only
	// There seems to be a YAML parsing issue with sigs.k8s.io/yaml
	// Only apply this workaround if SR_BENCHMARK_MODE or SR_TEST_MODE is set
	benchmarkMode := os.Getenv("SR_BENCHMARK_MODE")
	testMode := os.Getenv("SR_TEST_MODE")
	if (benchmarkMode == "1" || benchmarkMode == "true" || testMode == "1" || testMode == "true") &&
		!milvusConfig.Development.AutoCreateCollection && !milvusConfig.Development.DropCollectionOnStartup {
		logging.Warnf("Development settings parsed as false, forcing to true for benchmarks/tests\n")
		milvusConfig.Development.AutoCreateCollection = true
		milvusConfig.Development.DropCollectionOnStartup = true
	}

	// WORKAROUND: Force vector field settings if empty
	if milvusConfig.Collection.VectorField.Name == "" {
		logging.Warnf("VectorField.Name parsed as empty, setting to 'embedding'\n")
		milvusConfig.Collection.VectorField.Name = "embedding"
	}
	if milvusConfig.Collection.VectorField.MetricType == "" {
		logging.Warnf("VectorField.MetricType parsed as empty, setting to 'IP'\n")
		milvusConfig.Collection.VectorField.MetricType = "IP"
	}
	if milvusConfig.Collection.Index.Type == "" {
		logging.Warnf("Index.Type parsed as empty, setting to 'HNSW'\n")
		milvusConfig.Collection.Index.Type = "HNSW"
	}
	// Validate index params
	if milvusConfig.Collection.Index.Params.M == 0 {
		logging.Warnf("Index.Params.M parsed as 0, setting to 16\n")
		milvusConfig.Collection.Index.Params.M = 16
	}
	if milvusConfig.Collection.Index.Params.EfConstruction == 0 {
		logging.Warnf("Index.Params.EfConstruction parsed as 0, setting to 64\n")
		milvusConfig.Collection.Index.Params.EfConstruction = 64
	}
	// Validate search params
	if milvusConfig.Search.Params.Ef == 0 {
		logging.Warnf("Search.Params.Ef parsed as 0, setting to 64\n")
		milvusConfig.Search.Params.Ef = 64
	}

	return milvusConfig, nil
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
			{
				Name:     "ttl_seconds",
				DataType: entity.FieldTypeInt64,
			},
			{
				Name:     "expires_at",
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

// CheckConnection verifies the Milvus connection is healthy
func (c *MilvusCache) CheckConnection() error {
	if !c.enabled {
		return nil
	}

	if c.client == nil {
		return fmt.Errorf("milvus client is not initialized")
	}

	ctx := context.Background()
	if c.config != nil && c.config.Connection.Timeout > 0 {
		timeout := time.Duration(c.config.Connection.Timeout) * time.Second
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	// Simple connection check - list collections to verify connectivity
	// We don't check if specific collection exists here as it may not be created yet
	_, err := c.client.ListCollections(ctx)
	if err != nil {
		return fmt.Errorf("milvus connection check failed: %w", err)
	}

	return nil
}

// AddPendingRequest stores a request that is awaiting its response
func (c *MilvusCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Handle TTL=0: skip caching entirely
	if ttlSeconds == 0 {
		logging.Debugf("MilvusCache.AddPendingRequest: skipping cache (ttl_seconds=0)")
		return nil
	}

	// Store incomplete entry for later completion with response
	err := c.addEntry("", requestID, model, query, requestBody, nil, ttlSeconds)

	if err != nil {
		metrics.RecordCacheOperation("milvus", "add_pending", "error", time.Since(start).Seconds())
	} else {
		metrics.RecordCacheOperation("milvus", "add_pending", "success", time.Since(start).Seconds())
	}

	return err
}

// UpdateWithResponse completes a pending request by adding the response
func (c *MilvusCache) UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	logging.Debugf("MilvusCache.UpdateWithResponse: updating pending entry (request_id: %s, response_size: %d, ttl_seconds=%d)",
		requestID, len(responseBody), ttlSeconds)

	// Find the pending entry and complete it with the response
	// Query for the incomplete entry to retrieve its metadata
	ctx := context.Background()
	queryExpr := fmt.Sprintf("request_id == \"%s\" && response_body == \"\"", requestID)

	logging.Debugf("MilvusCache.UpdateWithResponse: searching for pending entry with expr: %s", queryExpr)

	// Note: We don't explicitly request "id" since Milvus auto-includes the primary key
	// We request model, query, request_body and will detect which column is which
	results, err := c.client.Query(ctx, c.collectionName, []string{}, queryExpr,
		[]string{"model", "query", "request_body"})
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

	// Milvus automatically includes the primary key in results but order is non-deterministic
	// We requested ["model", "query", "request_body"], expect 3-4 columns (primary key may be auto-included)
	// Strategy: Find the ID column (32-char hex string), then map remaining columns
	if len(results) < 3 {
		logging.Debugf("MilvusCache.UpdateWithResponse: unexpected result count: %d", len(results))
		metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("incomplete query result: expected 3+ columns, got %d", len(results))
	}

	var id, model, query, requestBody string
	idColIndex := -1

	// First pass: find the ID column (32-char hex string = MD5 hash)
	for i := 0; i < len(results); i++ {
		if col, ok := results[i].(*entity.ColumnVarChar); ok && col.Len() > 0 {
			val := col.Data()[0]
			if len(val) == 32 && isHexString(val) {
				id = val
				idColIndex = i
				break
			}
		}
	}

	// Second pass: extract data fields in order, skipping the ID column
	dataFieldIndex := 0
	for i := 0; i < len(results); i++ {
		if i == idColIndex {
			continue // Skip the primary key column
		}
		if col, ok := results[i].(*entity.ColumnVarChar); ok && col.Len() > 0 {
			val := col.Data()[0]
			switch dataFieldIndex {
			case 0:
				model = val
			case 1:
				query = val
			case 2:
				requestBody = val
			}
			dataFieldIndex++
		}
	}

	if id == "" || model == "" || query == "" {
		logging.Debugf("MilvusCache.UpdateWithResponse: failed to extract all required fields (id: %s, model: %s, query_len: %d)",
			id, model, len(query))
		metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to extract required fields from query result")
	}

	logging.Debugf("MilvusCache.UpdateWithResponse: found pending entry, adding complete entry (id: %s, model: %s)", id, model)

	// Create the complete entry with response data and TTL
	err = c.addEntry(id, requestID, model, query, []byte(requestBody), responseBody, ttlSeconds)
	if err != nil {
		metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to add complete entry: %w", err)
	}

	logging.Debugf("MilvusCache.UpdateWithResponse: successfully added complete entry with response")
	metrics.RecordCacheOperation("milvus", "update_response", "success", time.Since(start).Seconds())

	return nil
}

// AddEntry stores a complete request-response pair in the cache
func (c *MilvusCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Handle TTL=0: skip caching entirely
	if ttlSeconds == 0 {
		logging.Debugf("MilvusCache.AddEntry: skipping cache (ttl_seconds=0)")
		return nil
	}

	err := c.addEntry("", requestID, model, query, requestBody, responseBody, ttlSeconds)

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
func (c *MilvusCache) addEntry(id string, requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	// Determine effective TTL: use provided value or fall back to cache default
	effectiveTTL := ttlSeconds
	if ttlSeconds == -1 {
		effectiveTTL = c.ttlSeconds
	}

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

	now := time.Now()
	var expiresAt int64
	if effectiveTTL > 0 {
		expiresAt = now.Add(time.Duration(effectiveTTL) * time.Second).Unix()
	} else {
		expiresAt = 0 // No expiration
	}

	// Prepare data for upsert
	ids := []string{id}
	requestIDs := []string{requestID}
	models := []string{model}
	queries := []string{query}
	requestBodies := []string{string(requestBody)}
	responseBodies := []string{string(responseBody)}
	embeddings := [][]float32{embedding}
	timestamps := []int64{now.Unix()}
	ttlSecondsSlice := []int64{int64(effectiveTTL)}
	expiresAtSlice := []int64{expiresAt}

	// Create columns
	idColumn := entity.NewColumnVarChar("id", ids)
	requestIDColumn := entity.NewColumnVarChar("request_id", requestIDs)
	modelColumn := entity.NewColumnVarChar("model", models)
	queryColumn := entity.NewColumnVarChar("query", queries)
	requestColumn := entity.NewColumnVarChar("request_body", requestBodies)
	responseColumn := entity.NewColumnVarChar("response_body", responseBodies)
	embeddingColumn := entity.NewColumnFloatVector(c.config.Collection.VectorField.Name, len(embedding), embeddings)
	timestampColumn := entity.NewColumnInt64("timestamp", timestamps)
	ttlSecondsColumn := entity.NewColumnInt64("ttl_seconds", ttlSecondsSlice)
	expiresAtColumn := entity.NewColumnInt64("expires_at", expiresAtSlice)

	// Upsert the entry into the collection
	logging.Debugf("MilvusCache.addEntry: upserting entry into collection '%s' (embedding_dim: %d, request_size: %d, response_size: %d, ttl=%d)",
		c.collectionName, len(embedding), len(requestBody), len(responseBody), effectiveTTL)
	_, err = c.client.Upsert(ctx, c.collectionName, "", idColumn, requestIDColumn, modelColumn, queryColumn, requestColumn, responseColumn, embeddingColumn, timestampColumn, ttlSecondsColumn, expiresAtColumn)
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
		"ttl_seconds":         effectiveTTL,
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
	// Filter by has response and not expired (model filtering removed for cross-model cache sharing)
	now := time.Now().Unix()
	filterExpr := fmt.Sprintf("response_body != \"\" && (expires_at == 0 || expires_at > %d)", now)

	searchResult, err := c.client.Search(
		ctx,
		c.collectionName,
		[]string{},
		filterExpr,
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
		return nil, false, nil
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: no entries found")
		metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
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
		return nil, false, nil
	}

	// Cache Hit
	// Milvus automatically includes the primary key in search results but order is non-deterministic
	// Check which field is the response_body by detecting if field[0] is an MD5 hash
	responseBodyFieldIndex := 0
	if len(searchResult[0].Fields) > 1 {
		if testCol, ok := searchResult[0].Fields[0].(*entity.ColumnVarChar); ok && testCol.Len() > 0 {
			testVal := testCol.Data()[0]
			// If field[0] is exactly 32 hex chars, it's the ID hash, so response_body is in field[1]
			if len(testVal) == 32 && isHexString(testVal) {
				responseBodyFieldIndex = 1
			}
		}
	}

	var responseBody []byte
	responseBodyColumn, ok := searchResult[0].Fields[responseBodyFieldIndex].(*entity.ColumnVarChar)
	if ok && responseBodyColumn.Len() > 0 {
		responseBody = []byte(responseBodyColumn.Data()[0])
	}

	if responseBody == nil {
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: cache hit but response_body is missing or not a string")
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
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

	// Milvus automatically includes the primary key but column order may vary
	// We requested ["request_id", embedding_field], so we expect 2-3 columns
	// If 3 columns: primary key was auto-included, adjust indices
	requestIDColIndex := 0
	embeddingColIndex := 1
	expectedMinCols := 2

	if len(queryResult) >= 3 {
		// Primary key was auto-included, adjust indices
		requestIDColIndex = 1
		embeddingColIndex = 2
	}

	if len(queryResult) < expectedMinCols {
		logging.Infof("MilvusCache.GetAllEntries: no entries found or incomplete result")
		return []string{}, [][]float32{}, nil
	}

	// Extract request IDs
	requestIDColumn, ok := queryResult[requestIDColIndex].(*entity.ColumnVarChar)
	if !ok {
		return nil, nil, fmt.Errorf("unexpected request_id column type: %T", queryResult[requestIDColIndex])
	}

	// Extract embeddings
	embeddingColumn, ok := queryResult[embeddingColIndex].(*entity.ColumnFloatVector)
	if !ok {
		return nil, nil, fmt.Errorf("unexpected embedding column type: %T", queryResult[embeddingColIndex])
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

// isHexString checks if a string contains only hexadecimal characters
func isHexString(s string) bool {
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') && (c < 'A' || c > 'F') {
			return false
		}
	}
	return true
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

	// Milvus automatically includes the primary key but the column order is non-deterministic
	// We need to find which column is the response_body by checking which is NOT the primary key (32-char hash)
	responseBodyColIndex := 0
	if len(queryResult) > 1 {
		// Check if column[0] looks like an MD5 hash (32 hex chars)
		if testCol, ok := queryResult[0].(*entity.ColumnVarChar); ok && testCol.Len() > 0 {
			testVal, _ := testCol.ValueByIdx(0)
			// If it's exactly 32 chars and all hex, it's likely the ID hash
			if len(testVal) == 32 && isHexString(testVal) {
				responseBodyColIndex = 1 // response_body is in column 1
			} else {
				responseBodyColIndex = 0 // response_body is in column 0
			}
		}
	}

	// Extract response body
	responseBodyColumn, ok := queryResult[responseBodyColIndex].(*entity.ColumnVarChar)
	if !ok {
		logging.Debugf("MilvusCache.GetByID: unexpected response_body column type: %T", queryResult[responseBodyColIndex])
		metrics.RecordCacheOperation("milvus", "get_by_id", "error", time.Since(start).Seconds())
		return nil, fmt.Errorf("invalid response_body column type: %T", queryResult[responseBodyColIndex])
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

// SearchDocuments performs vector search on a specified collection for RAG retrieval
// This method is used by the RAG plugin to retrieve context from knowledge bases
//
// Parameters:
//   - vectorFieldName: Name of the vector field in the collection (defaults to cache config)
//   - metricType: Metric type for similarity search (defaults to cache config)
//   - ef: HNSW search parameter ef (defaults to cache config)
//
// If these parameters are empty/zero, the method uses the cache collection's configuration.
// This allows RAG collections to use different configurations when needed.
func (c *MilvusCache) SearchDocuments(ctx context.Context, collectionName string, queryEmbedding []float32, threshold float32, topK int, filterExpr string, contentField string, vectorFieldName string, metricType string, ef int) ([]string, []float32, error) {
	if !c.enabled {
		return nil, nil, fmt.Errorf("milvus cache is not enabled")
	}

	if c.client == nil {
		return nil, nil, fmt.Errorf("milvus client is not initialized")
	}

	// Use provided parameters or fall back to cache config defaults
	actualVectorFieldName := vectorFieldName
	if actualVectorFieldName == "" {
		actualVectorFieldName = c.config.Collection.VectorField.Name
	}

	actualMetricType := metricType
	if actualMetricType == "" {
		actualMetricType = c.config.Collection.VectorField.MetricType
	}

	actualEf := ef
	if actualEf == 0 {
		actualEf = c.config.Search.Params.Ef
	}

	// Define search parameters
	searchParam, err := entity.NewIndexHNSWSearchParam(actualEf)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create search parameters: %w", err)
	}

	// Build filter expression
	// If no filter provided and contentField is specified, default to filtering for non-empty content
	if filterExpr == "" && contentField != "" {
		filterExpr = fmt.Sprintf("%s != \"\"", contentField)
	}

	// Use Milvus Search with collection-specific or default parameters
	searchResult, err := c.client.Search(
		ctx,
		collectionName,
		[]string{},
		filterExpr,
		[]string{contentField},
		[]entity.Vector{entity.FloatVector(queryEmbedding)},
		actualVectorFieldName,
		entity.MetricType(actualMetricType),
		topK,
		searchParam,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("milvus search failed: %w", err)
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		return nil, nil, nil // No results, but not an error
	}

	// Extract results
	var contents []string
	var scores []float32

	for i := 0; i < searchResult[0].ResultCount; i++ {
		score := searchResult[0].Scores[i]
		if score < threshold {
			continue // Skip results below threshold
		}

		// Extract content from result
		// Milvus may include the primary key field even when we only request one field,
		// so we need to find the contentField by checking field types and values.
		// We iterate through fields to find the VarChar column that matches our contentField.
		var content string
		found := false

		for _, field := range searchResult[0].Fields {
			if contentCol, ok := field.(*entity.ColumnVarChar); ok {
				// Check if this column has enough entries and get the value
				if contentCol.Len() > i {
					fieldValue, err := contentCol.ValueByIdx(i)
					if err == nil && fieldValue != "" {
						// If we requested only one field, assume it's the content field
						// Otherwise, we'd need to match by field name (not available in Milvus API)
						// For now, since we only request contentField, the first VarChar field
						// that's not a 32-char hex string (likely an ID) should be our content
						if len(fieldValue) != 32 || !isHexString(fieldValue) {
							content = fieldValue
							found = true
							break
						}
					}
				}
			}
		}

		if found && content != "" {
			contents = append(contents, content)
			scores = append(scores, score)
		} else {
			// Fallback: if we couldn't find content, log a warning but continue
			// This shouldn't happen if the collection is properly configured
			logging.Warnf("SearchDocuments: could not extract content for result %d (score=%.3f)", i, score)
		}
	}

	return contents, scores, nil
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
