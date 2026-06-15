package cache

import (
	"context"
	"crypto/md5"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"sigs.k8s.io/yaml"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// MilvusCache provides a scalable semantic cache implementation using Milvus vector database
type MilvusCache struct {
	SimilarityTracker   // embedded — provides LastSimilarity()
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
	embeddingModel      string // "bert", "qwen3", "gemma", "mmbert", or "multimodal"
}

// MilvusCacheOptions contains configuration parameters for Milvus cache initialization
type MilvusCacheOptions struct {
	SimilarityThreshold float32
	TTLSeconds          int
	Enabled             bool
	Config              *config.MilvusConfig
	ConfigPath          string
	EmbeddingModel      string
}

// NewMilvusCache initializes a new Milvus-backed semantic cache instance
//
//nolint:funlen
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
	logging.Debugf("MilvusCache: config loaded - host=%s:%d, collection=%s, dimension=%d",
		milvusConfig.Connection.Host, milvusConfig.Connection.Port, milvusConfig.Collection.Name,
		semanticCacheEmbeddingDimension(milvusConfig.Collection.VectorField.Dimension, options.EmbeddingModel))

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
	milvusClient, err := milvuslifecycle.ConnectGRPC(dialCtx, connectionString, 0)
	if err != nil {
		logging.Debugf("MilvusCache: failed to connect: %v", err)
		return nil, fmt.Errorf("failed to create Milvus client: %w", err)
	}

	embeddingModel := normalizeEmbeddingModel(options.EmbeddingModel)

	cache := &MilvusCache{
		client:              milvusClient,
		config:              milvusConfig,
		collectionName:      milvusConfig.Collection.Name,
		similarityThreshold: options.SimilarityThreshold,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
		embeddingModel:      embeddingModel,
	}

	// Test connection using the new CheckConnection method
	if err := cache.CheckConnection(); err != nil {
		logging.Debugf("MilvusCache: connection check failed: %v", err)
		_ = milvusClient.Close() // best-effort close
		return nil, err
	}
	logging.Debugf("MilvusCache: successfully connected to Milvus")

	// Set up the collection for caching
	logging.Debugf("MilvusCache: initializing collection '%s'", milvusConfig.Collection.Name)
	if err := cache.initializeCollection(); err != nil {
		logging.Debugf("MilvusCache: failed to initialize collection: %v", err)
		_ = milvusClient.Close() // best-effort close
		return nil, fmt.Errorf("failed to initialize collection: %w", err)
	}
	logging.Debugf("MilvusCache: initialization complete")

	return cache, nil
}

// loadMilvusConfig reads and parses the Milvus configuration from file (Deprecated)
//
//nolint:cyclop,funlen
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
		logging.Debugf("MilvusCache: dropped existing collection '%s' for development", c.collectionName)
		logging.LogEvent("collection_dropped", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"reason":     "development_mode",
		})
	}

	if err := milvuslifecycle.EnsureCollectionLoadedWithRetry(ctx, c.client, c.collectionName, func(innerCtx context.Context) error {
		logging.Debugf("MilvusCache: collection '%s' does not exist. AutoCreateCollection=%v",
			c.collectionName, c.config.Development.AutoCreateCollection)
		if !c.config.Development.AutoCreateCollection {
			return fmt.Errorf("collection %s does not exist and auto-creation is disabled", c.collectionName)
		}
		if err := c.createCollection(innerCtx); err != nil {
			return err
		}
		logging.Debugf("MilvusCache: created new collection '%s' with dimension %d",
			c.collectionName, c.config.Collection.VectorField.Dimension)
		logging.LogEvent("collection_created", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"dimension":  c.config.Collection.VectorField.Dimension,
		})
		return nil
	}, milvuslifecycle.CollectionRetryOptions{}); err != nil {
		logging.Debugf("MilvusCache: failed to ensure/load collection: %v", err)
		return fmt.Errorf("failed to ensure/load collection: %w", err)
	}

	return nil
}

// getEmbedding generates an embedding based on the configured embedding model
func (c *MilvusCache) getEmbedding(text string) ([]float32, error) {
	modelName := c.embeddingModel

	switch modelName {
	case "qwen3":
		// Use GetEmbeddingBatched for Qwen3 with batching support
		output, err := candle_binding.GetEmbeddingBatched(text, modelName, c.embeddingDimension())
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "gemma":
		// Use GetEmbeddingWithModelType for Gemma
		output, err := candle_binding.GetEmbeddingWithModelType(text, modelName, c.embeddingDimension())
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "mmbert":
		output, err := candle_binding.GetEmbeddingWithModelType(text, modelName, c.embeddingDimension())
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "multimodal":
		output, err := candle_binding.GetEmbeddingWithModelType(text, modelName, c.embeddingDimension())
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "bert":
		// Use traditional GetEmbedding for BERT (default)
		return candle_binding.GetEmbedding(text, 0)
	default:
		return nil, fmt.Errorf("unsupported embedding model: %s (must be 'bert', 'qwen3', 'gemma', 'mmbert', or 'multimodal')", c.embeddingModel)
	}
}

func (c *MilvusCache) embeddingDimension() int {
	if c == nil || c.config == nil {
		return semanticCacheEmbeddingDimension(0, "")
	}
	return semanticCacheEmbeddingDimension(c.config.Collection.VectorField.Dimension, c.embeddingModel)
}

// createCollection builds the Milvus collection with the appropriate schema
func (c *MilvusCache) createCollection(ctx context.Context) error {
	actualDimension := c.embeddingDimension()
	c.config.Collection.VectorField.Dimension = actualDimension

	logging.Debugf("MilvusCache.createCollection: using embedding dimension: %d", actualDimension)

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
					"dim": fmt.Sprintf("%d", actualDimension),
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
//
//nolint:gocognit,cyclop,funlen
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
//
//nolint:funlen
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
		embedding, err := c.getEmbedding(entry.Query)
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
//
//nolint:funlen
func (c *MilvusCache) addEntry(id string, requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error {
	// Determine effective TTL: use provided value or fall back to cache default
	effectiveTTL := ttlSeconds
	if ttlSeconds == -1 {
		effectiveTTL = c.ttlSeconds
	}

	// Generate semantic embedding for the query
	embedding, err := c.getEmbedding(query)
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
