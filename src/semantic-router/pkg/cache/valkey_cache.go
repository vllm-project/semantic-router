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
)

// ValkeyCache provides a scalable semantic cache implementation using Valkey with vector search
type ValkeyCache struct {
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

	// Validate that config is provided
	if options.Config == nil {
		return nil, fmt.Errorf("valkey config is required")
	}

	valkeyConfig := options.Config
	logging.Debugf("ValkeyCache: config loaded - host=%s:%d, index=%s, dimension=auto-detect",
		valkeyConfig.Connection.Host, valkeyConfig.Connection.Port, valkeyConfig.Index.Name)

	// Establish connection to Valkey server
	logging.Debugf("ValkeyCache: connecting to Valkey at %s:%d", valkeyConfig.Connection.Host, valkeyConfig.Connection.Port)

	// Create Valkey client configuration
	clientConfig := config.NewClientConfiguration().
		WithAddress(&config.NodeAddress{
			Host: valkeyConfig.Connection.Host,
			Port: valkeyConfig.Connection.Port,
		})

	// Add credentials if password is configured
	if valkeyConfig.Connection.Password != "" {
		clientConfig = clientConfig.WithCredentials(
			config.NewServerCredentials("", valkeyConfig.Connection.Password),
		)
	}

	// Add database selection if not default
	if valkeyConfig.Connection.Database != 0 {
		clientConfig = clientConfig.WithDatabaseId(valkeyConfig.Connection.Database)
	}

	// Add timeout if configured (in milliseconds)
	if valkeyConfig.Connection.Timeout > 0 {
		timeout := time.Duration(valkeyConfig.Connection.Timeout) * time.Second
		clientConfig = clientConfig.WithRequestTimeout(timeout)
	}

	// Create the Valkey client
	valkeyClient, err := glide.NewClient(clientConfig)
	if err != nil {
		logging.Debugf("ValkeyCache: failed to create client: %v", err)
		return nil, fmt.Errorf("failed to create Valkey client: %w", err)
	}

	// Default to "bert" if no embedding model specified
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

	// Test connection using the new CheckConnection method
	if err := cache.CheckConnection(); err != nil {
		logging.Debugf("ValkeyCache: failed to connect: %v", err)
		return nil, err
	}
	logging.Debugf("ValkeyCache: successfully connected to Valkey")

	// Set up the index for vector search
	logging.Debugf("ValkeyCache: initializing index '%s'", valkeyConfig.Index.Name)
	if err := cache.initializeIndex(); err != nil {
		logging.Debugf("ValkeyCache: failed to initialize index: %v", err)
		valkeyClient.Close()
		return nil, fmt.Errorf("failed to initialize index: %w", err)
	}
	logging.Debugf("ValkeyCache: initialization complete")

	return cache, nil
}

// initializeIndex sets up the Valkey index for vector search
func (c *ValkeyCache) initializeIndex() error {
	ctx := context.Background()

	// Check if index exists using FT.INFO
	_, err := c.client.CustomCommand(ctx, []string{"FT.INFO", c.indexName})
	indexExists := err == nil

	// Handle development mode index reset
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

	// Create index if it doesn't exist
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

	// Determine embedding dimension automatically
	testEmbedding, err := c.getEmbedding("test")
	if err != nil {
		return fmt.Errorf("failed to detect embedding dimension: %w", err)
	}
	actualDimension := len(testEmbedding)

	logging.Debugf("ValkeyCache.createIndex: auto-detected embedding dimension: %d", actualDimension)

	// Determine distance metric for Valkey
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

	// Build FT.CREATE command using executeCommand
	var createCmd []string
	if c.config.Index.IndexType == "HNSW" {
		createCmd = []string{
			"FT.CREATE", c.indexName,
			"ON", "HASH",
			"PREFIX", "1", c.config.Index.Prefix,
			"SCHEMA",
			"request_id", "TAG", // Changed from TEXT to TAG for exact matching
			"model", "TAG",
			"query", "TEXT",
			"request_body", "TEXT",
			"response_body", "TEXT",
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
			"request_id", "TAG", // Changed from TEXT to TAG for exact matching
			"model", "TAG",
			"query", "TEXT",
			"request_body", "TEXT",
			"response_body", "TEXT",
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

	// Handle TTL=0: skip caching entirely
	if ttlSeconds == 0 {
		logging.Debugf("ValkeyCache.AddPendingRequest: skipping cache (ttl_seconds=0)")
		return nil
	}

	// Store incomplete entry for later completion with response
	err := c.addEntry("", requestID, model, query, requestBody, nil, ttlSeconds)

	if err != nil {
		metrics.RecordCacheOperation("valkey", "add_pending", "error", time.Since(start).Seconds())
	} else {
		metrics.RecordCacheOperation("valkey", "add_pending", "success", time.Since(start).Seconds())
	}

	return err
}

// UpdateWithResponse completes a pending request by adding the response
// pendingEntry holds the parsed fields from a pending cache entry search result.
type pendingEntry struct {
	docID          string
	model          string
	query          string
	requestBodyStr string
}

// parsePendingSearchResult extracts a pendingEntry from a Valkey FT.SEARCH result.
// Returns the entry or an error describing the parse failure.
// extractPendingFields extracts a pendingEntry from a Valkey GLIDE doc map (the second element of FT.SEARCH results).
func extractPendingFields(docMap map[string]interface{}) (*pendingEntry, error) {
	entry := &pendingEntry{}
	for docKey, docValue := range docMap {
		entry.docID = docKey

		fieldsMap, mapOk := docValue.(map[string]interface{})
		if !mapOk {
			logging.Warnf("UpdateWithResponse: document fields is not a map, type=%T", docValue)
			return nil, fmt.Errorf("invalid search result: expected fields map")
		}

		if v, exists := fieldsMap["model"]; exists {
			entry.model = fmt.Sprint(v)
		}
		if v, exists := fieldsMap["query"]; exists {
			entry.query = fmt.Sprint(v)
		}
		if v, exists := fieldsMap["request_body"]; exists {
			entry.requestBodyStr = fmt.Sprint(v)
		}

		break // Only process the first document
	}
	return entry, nil
}

func parsePendingSearchResult(results interface{}, requestID string, prefix string) (*pendingEntry, error) {
	resultsArray, ok := results.([]interface{})
	if !ok || len(resultsArray) < 1 {
		logging.Infof("ValkeyCache.UpdateWithResponse: invalid result format for request_id=%s", requestID)
		return nil, fmt.Errorf("invalid search result format")
	}

	totalResults, ok := resultsArray[0].(int64)
	if !ok {
		logging.Infof("ValkeyCache.UpdateWithResponse: invalid count type for request_id=%s (got %T)", requestID, resultsArray[0])
		return nil, fmt.Errorf("invalid search result count type")
	}

	if totalResults == 0 {
		logging.Infof("ValkeyCache.UpdateWithResponse: no pending entry found with request_id=%s (count=0, may still be indexing)", requestID)
		return nil, fmt.Errorf("no pending entry found (indexing may still be in progress)")
	}

	logging.Infof("UpdateWithResponse: found %d result(s) for request_id=%s", totalResults, requestID)

	if len(resultsArray) < 2 {
		logging.Warnf("UpdateWithResponse: resultsArray only has %d elements", len(resultsArray))
		return nil, fmt.Errorf("invalid search result: expected at least 2 elements")
	}

	docMap, ok := resultsArray[1].(map[string]interface{})
	if !ok {
		logging.Warnf("UpdateWithResponse: resultsArray[1] is not a map, type=%T", resultsArray[1])
		return nil, fmt.Errorf("invalid search result: expected map at index 1")
	}

	entry, err := extractPendingFields(docMap)
	if err != nil {
		return nil, err
	}

	if !strings.HasPrefix(entry.docID, prefix) {
		logging.Warnf("UpdateWithResponse: docID '%s' doesn't have expected prefix '%s'", entry.docID, prefix)
	}

	logging.Debugf("UpdateWithResponse: extracted docID='%s', model='%s', query='%s'", entry.docID, entry.model, entry.query)

	if entry.model == "" || entry.query == "" {
		logging.Warnf("UpdateWithResponse: missing required fields (model='%s', query='%s')", entry.model, entry.query)
		return nil, fmt.Errorf("missing required fields in pending entry")
	}

	return entry, nil
}

func (c *ValkeyCache) UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	logging.Debugf("ValkeyCache.UpdateWithResponse: updating pending entry (request_id: %s, response_size: %d, ttl_seconds=%d)",
		requestID, len(responseBody), ttlSeconds)

	// Find the pending entry by request_id
	ctx := context.Background()

	// Search for documents with matching request_id
	// TAG field syntax: @field:{value}
	query := fmt.Sprintf("@request_id:{%s}", requestID)
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

	// Handle TTL=0: skip caching entirely
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

	// Convert embedding to bytes
	embeddingBytes := floatsToBytes(embedding)

	// Prepare document key with prefix (check if already prefixed to avoid double prefix)
	var docKey string
	if strings.HasPrefix(id, c.config.Index.Prefix) {
		docKey = id // Already has prefix, use as-is
	} else {
		docKey = c.config.Index.Prefix + id // Add prefix
	}

	// Store as Valkey hash using HSET
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

	// Set TTL if configured (Valkey native TTL)
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

// searchMatch holds the parsed fields from a vector search result.
type searchMatch struct {
	distance     float64
	responseBody interface{}
}

// parseBestMatch extracts the best-match distance and response body from a Valkey FT.SEARCH vector result.
// Returns nil when no valid match is found.
func parseBestMatch(searchResult interface{}) *searchMatch {
	resultsArray, ok := searchResult.([]interface{})
	if !ok || len(resultsArray) < 1 {
		return nil
	}

	totalResults, ok := resultsArray[0].(int64)
	if !ok || totalResults == 0 {
		return nil
	}

	if len(resultsArray) < 2 {
		return nil
	}

	docMap, ok := resultsArray[1].(map[string]interface{})
	if !ok {
		logging.Debugf("ValkeyCache.FindSimilarWithThreshold: invalid result format, expected map at index 1, got %T", resultsArray[1])
		return nil
	}

	for _, docValue := range docMap {
		fieldsMap, mapOk := docValue.(map[string]interface{})
		if !mapOk {
			logging.Debugf("ValkeyCache.FindSimilarWithThreshold: invalid fields format, expected map, got %T", docValue)
			return nil
		}

		match := &searchMatch{}
		distanceVal, exists := fieldsMap["vector_distance"]
		if !exists {
			return nil
		}

		if _, err := fmt.Sscanf(fmt.Sprint(distanceVal), "%f", &match.distance); err != nil {
			logging.Debugf("ValkeyCache.FindSimilarWithThreshold: failed to parse distance value: %v", err)
			return nil
		}

		match.responseBody = fieldsMap["response_body"]
		return match // Only process the first (best) document
	}

	return nil
}

// distanceToSimilarity converts a vector distance to a similarity score based on the metric type.
func distanceToSimilarity(metricType string, distance float64) float32 {
	switch metricType {
	case "COSINE":
		return 1.0 - float32(distance)/2.0
	case "IP":
		return float32(distance)
	case "L2":
		return 1.0 / (1.0 + float32(distance))
	default:
		return 1.0 - float32(distance)
	}
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

// extractResponseBody returns the response bytes from a search match, or nil if missing/empty.
func extractResponseBody(match *searchMatch) []byte {
	if match.responseBody == nil {
		return nil
	}
	s := fmt.Sprint(match.responseBody)
	if s == "" {
		return nil
	}
	return []byte(s)
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *ValkeyCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	start := time.Now()

	if !c.enabled {
		return nil, false, nil
	}

	// Generate semantic embedding for similarity comparison
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
