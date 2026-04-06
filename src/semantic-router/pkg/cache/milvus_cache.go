package cache

import (
	"context"
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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvuslifecycle"
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
func NewMilvusCache(options MilvusCacheOptions) (*MilvusCache, error) {
	if !options.Enabled {
		logging.Debugf("MilvusCache: disabled, returning stub")
		return &MilvusCache{enabled: false}, nil
	}

	milvusConfig, err := resolveMilvusConfig(options)
	if err != nil {
		logging.Debugf("MilvusCache: failed to load config: %v", err)
		return nil, err
	}
	logging.Debugf("MilvusCache: config loaded - host=%s:%d, collection=%s, dimension=auto-detect",
		milvusConfig.Connection.Host, milvusConfig.Connection.Port, milvusConfig.Collection.Name)

	milvusClient, err := connectMilvusClient(milvusConfig)
	if err != nil {
		logging.Debugf("MilvusCache: failed to connect: %v", err)
		return nil, err
	}

	cache := &MilvusCache{
		client:              milvusClient,
		config:              milvusConfig,
		collectionName:      milvusConfig.Collection.Name,
		similarityThreshold: options.SimilarityThreshold,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
		embeddingModel:      defaultMilvusEmbeddingModel(options.EmbeddingModel),
	}

	if err := cache.CheckConnection(); err != nil {
		logging.Debugf("MilvusCache: connection check failed: %v", err)
		closeMilvusClient(milvusClient, "connection check")
		return nil, err
	}
	logging.Debugf("MilvusCache: successfully connected to Milvus")

	logging.Debugf("MilvusCache: initializing collection '%s'", milvusConfig.Collection.Name)
	if err := cache.initializeCollection(); err != nil {
		logging.Debugf("MilvusCache: failed to initialize collection: %v", err)
		closeMilvusClient(milvusClient, "collection initialization")
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
	if err := yaml.Unmarshal(data, &milvusConfig); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	logParsedMilvusConfig(configPath, milvusConfig)
	applyMilvusConfigEnvironmentOverrides(milvusConfig)
	applyMilvusConfigDefaults(milvusConfig)
	return milvusConfig, nil
}

// initializeCollection sets up the Milvus collection and index structures
func (c *MilvusCache) initializeCollection() error {
	ctx := context.Background()
	spec, err := c.collectionSpec()
	if err != nil {
		return err
	}

	result, err := milvuslifecycle.EnsureCollection(ctx, c.client, spec, milvuslifecycle.EnsureOptions{
		AllowCreate:  c.config.Development.AutoCreateCollection,
		DropExisting: c.config.Development.DropCollectionOnStartup,
	})
	if err != nil {
		logging.Debugf("MilvusCache: failed to initialize collection lifecycle: %v", err)
		return err
	}

	if result.Dropped {
		logging.Debugf("MilvusCache: dropped existing collection '%s' for development", c.collectionName)
		logging.LogEvent("collection_dropped", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"reason":     "development_mode",
		})
	}
	if result.Created {
		logging.Debugf("MilvusCache: created new collection '%s' with dimension %d",
			c.collectionName, c.config.Collection.VectorField.Dimension)
		logging.LogEvent("collection_created", map[string]interface{}{
			"backend":    "milvus",
			"collection": c.collectionName,
			"dimension":  c.config.Collection.VectorField.Dimension,
		})
	}
	logging.Debugf("MilvusCache: collection loaded successfully")

	return nil
}

// getEmbedding generates an embedding based on the configured embedding model
func (c *MilvusCache) getEmbedding(text string) ([]float32, error) {
	modelName := c.embeddingModel
	if modelName == "" {
		modelName = "bert"
	}

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

// collectionSpec builds the shared Milvus lifecycle spec for the cache collection.
func (c *MilvusCache) collectionSpec() (milvuslifecycle.CollectionSpec, error) {
	// Determine embedding dimension automatically
	testEmbedding, err := c.getEmbedding("test")
	if err != nil {
		return milvuslifecycle.CollectionSpec{}, fmt.Errorf("failed to detect embedding dimension: %w", err)
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

	return milvuslifecycle.CollectionSpec{
		Name:     c.collectionName,
		Schema:   schema,
		ShardNum: 1,
		Load:     true,
		Indexes: []milvuslifecycle.IndexSpec{
			{
				FieldName: c.config.Collection.VectorField.Name,
				Build: func() (entity.Index, error) {
					return entity.NewIndexHNSW(
						entity.MetricType(c.config.Collection.VectorField.MetricType),
						c.config.Collection.Index.Params.M,
						c.config.Collection.Index.Params.EfConstruction,
					)
				},
			},
		},
	}, nil
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

	pendingEntry, err := c.loadPendingEntry(requestID)
	if err != nil {
		logging.Debugf("MilvusCache.UpdateWithResponse: lookup failed: %v", err)
		metrics.RecordCacheOperation("milvus", "update_response", "error", time.Since(start).Seconds())
		return err
	}

	logging.Debugf("MilvusCache.UpdateWithResponse: found pending entry, adding complete entry (id: %s, model: %s)",
		pendingEntry.ID, pendingEntry.Model)
	if err := c.addEntry(pendingEntry.ID, requestID, pendingEntry.Model, pendingEntry.Query, []byte(pendingEntry.RequestBody), responseBody, ttlSeconds); err != nil {
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

	if !c.enabled || len(entries) == 0 {
		return nil
	}

	logging.Debugf("MilvusCache.AddEntriesBatch: adding %d entries in batch", len(entries))
	batchData, err := c.buildBatchUpsertData(entries)
	if err != nil {
		return err
	}

	columns := buildMilvusBatchColumns(batchData, c.config.Collection.VectorField.Name)
	logging.Debugf("MilvusCache.AddEntriesBatch: upserting %d entries into collection '%s'",
		len(entries), c.collectionName)
	if _, err := c.client.Upsert(context.Background(), c.collectionName, "", columns...); err != nil {
		logging.Debugf("MilvusCache.AddEntriesBatch: upsert failed: %v", err)
		metrics.RecordCacheOperation("milvus", "add_entries_batch", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to upsert cache entries: %w", err)
	}

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
	upsertData, err := c.buildSingleUpsertData(id, requestID, model, query, requestBody, responseBody, ttlSeconds)
	if err != nil {
		return err
	}

	columns := buildMilvusSingleColumns(upsertData, c.config.Collection.VectorField.Name)
	logging.Debugf("MilvusCache.addEntry: upserting entry into collection '%s' (embedding_dim: %d, request_size: %d, response_size: %d, ttl=%d)",
		c.collectionName, len(upsertData.embedding), len(requestBody), len(responseBody), upsertData.ttlSeconds)
	if _, err := c.client.Upsert(context.Background(), c.collectionName, "", columns...); err != nil {
		logging.Debugf("MilvusCache.addEntry: upsert failed: %v", err)
		return fmt.Errorf("failed to upsert cache entry: %w", err)
	}

	if err := c.client.Flush(context.Background(), c.collectionName, false); err != nil {
		logging.Warnf("Failed to flush cache entry: %v", err)
	}

	logging.Debugf("MilvusCache.addEntry: successfully added entry to Milvus")
	logging.LogEvent("cache_entry_added", map[string]interface{}{
		"backend":             "milvus",
		"collection":          c.collectionName,
		"request_id":          requestID,
		"query":               query,
		"model":               model,
		"embedding_dimension": len(upsertData.embedding),
		"ttl_seconds":         upsertData.ttlSeconds,
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

	queryEmbedding, err := c.getEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	searchParam, err := entity.NewIndexHNSWSearchParam(c.config.Search.Params.Ef)
	if err != nil {
		return nil, false, fmt.Errorf("failed to create search parameters: %w", err)
	}

	filterExpr := fmt.Sprintf("response_body != \"\" && (expires_at == 0 || expires_at > %d)", time.Now().Unix())
	searchResult, err := c.client.Search(
		context.Background(),
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
	c.StoreSimilarity(bestScore)
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

	responseBody, err := extractMilvusResponseBody(searchResult[0].Fields, 0)
	if err != nil {
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: cache hit but response_body is missing: %v", err)
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
	queryResult, err := c.client.Query(
		ctx,
		c.collectionName,
		[]string{},
		fmt.Sprintf("request_id == \"%s\" && response_body != \"\"", requestID),
		[]string{"response_body"},
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

	responseBody, err := extractMilvusResponseBody(queryResult, 0)
	if err != nil {
		logging.Debugf("MilvusCache.GetByID: response_body extraction failed: %v", err)
		metrics.RecordCacheOperation("milvus", "get_by_id", "error", time.Since(start).Seconds())
		return nil, fmt.Errorf("failed to extract response_body: %w", err)
	}
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

	searchParam, err := entity.NewIndexHNSWSearchParam(actualEf)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create search parameters: %w", err)
	}
	if filterExpr == "" && contentField != "" {
		filterExpr = fmt.Sprintf("%s != \"\"", contentField)
	}

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
		return nil, nil, nil
	}

	var contents []string
	var scores []float32
	for i := 0; i < searchResult[0].ResultCount; i++ {
		score := searchResult[0].Scores[i]
		if score < threshold {
			continue
		}

		content, found := extractSearchDocumentContent(searchResult[0].Fields, i)
		if !found || content == "" {
			logging.Warnf("SearchDocuments: could not extract content for result %d (score=%.3f)", i, score)
			continue
		}

		contents = append(contents, content)
		scores = append(scores, score)
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

	hitRatio := 0.0
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	cacheStats := CacheStats{
		TotalEntries: c.collectionRowCount(),
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}
	if c.lastCleanupTime != nil {
		cacheStats.LastCleanupTime = c.lastCleanupTime
	}
	return cacheStats
}
