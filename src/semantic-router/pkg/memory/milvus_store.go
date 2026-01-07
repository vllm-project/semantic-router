package memory

import (
	"context"
	"fmt"
	"strings"
	"time"
	"encoding/json"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DefaultMaxRetries is the default number of retry attempts for transient errors
// DefaultRetryBaseDelay is the base delay for exponential backoff (in milliseconds)
const (
	DefaultMaxRetries = 3
	DefaultRetryBaseDelay = 100
)

// MilvusStore provides memory retrieval from Milvus with similarity threshold filtering
type MilvusStore struct {
	client         client.Client
	collectionName string
	config         config.MemoryConfig
	enabled        bool
	maxRetries     int
	retryBaseDelay time.Duration
}

// MilvusStoreOptions contains configuration for creating a MilvusStore
// 	Client is the Milvus client instance
// 	CollectionName is the name of the Milvus collection
// 	Config is the memory configuration
// 	Enabled controls whether the store is active
type MilvusStoreOptions struct {
	Client client.Client
	CollectionName string
	Config config.MemoryConfig
	Enabled bool
}

// NewMilvusStore creates a new MilvusStore instance
func NewMilvusStore(options MilvusStoreOptions) (*MilvusStore, error) {
	if !options.Enabled {
		logging.Debugf("MilvusStore: disabled, returning stub")
		return &MilvusStore{
			enabled: false,
		}, nil
	}

	if options.Client == nil {
		return nil, fmt.Errorf("Milvus client is required")
	}

	if options.CollectionName == "" {
		return nil, fmt.Errorf("Collection name is required")
	}

	// Use default config if not provided
	config := options.Config
	if config.Embedding.Model == "" {
		config = DefaultMemoryConfig()
	}

	store := &MilvusStore{
		client:         options.Client,
		collectionName: options.CollectionName,
		config:         config,
		enabled:        options.Enabled,
		maxRetries:     DefaultMaxRetries,
		retryBaseDelay: DefaultRetryBaseDelay * time.Millisecond,
	}

	logging.Debugf("MilvusStore: initialized with collection='%s', model='%s', dimension=%d",
		store.collectionName, store.config.Embedding.Model, store.config.Embedding.Dimension)

	return store, nil
}

// Retrieve searches for memories in Milvus with similarity threshold filtering
func (m *MilvusStore) Retrieve(ctx context.Context, opts RetrieveOptions) ([]RetrieveResult, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}

	// Apply defaults
	limit := opts.Limit
	if limit <= 0 {
		limit = m.config.DefaultRetrievalLimit
	}

	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = m.config.DefaultSimilarityThreshold
	}

	if opts.Query == "" {
		return nil, fmt.Errorf("Query is required")
	}

	if opts.UserID == "" {
		return nil, fmt.Errorf("User id is required")
	}

	logging.Debugf("MilvusStore.Retrieve: query='%s', user_id='%s', limit=%d, threshold=%.4f",
		opts.Query, opts.UserID, limit, threshold)

	// Generate embedding for the query using all-MiniLM-L6-v2 (384-dim)
	// Note: The candle_binding.GetEmbedding uses the initialized model
	// For all-MiniLM-L6-v2, we need to ensure it's initialized
	embedding, err := candle_binding.GetEmbedding(opts.Query, 512)
	if err != nil {
		return nil, fmt.Errorf("Failed to generate embedding: %w", err)
	}

	// Validate embedding dimension matches expected
	expectedDim := m.config.Embedding.Dimension
	if expectedDim == 0 {
		expectedDim = 384 // Default for all-MiniLM-L6-v2
	}

	if len(embedding) != expectedDim {
		logging.Warnf("MilvusStore.Retrieve: embedding dimension mismatch - got %d, expected %d",
			len(embedding), expectedDim)
		// Continue anyway, but log the warning
	}

	logging.Debugf("MilvusStore.Retrieve: generated embedding with dimension %d", len(embedding))

	// Build filter expression for user_id
	filterExpr := fmt.Sprintf("user_id == \"%s\"", opts.UserID)

	// Add memory type filter if specified
	if len(opts.Types) > 0 {
		typeFilter := "("
		for i, memType := range opts.Types {
			if i > 0 {
				typeFilter += " || "
			}
			typeFilter += fmt.Sprintf("memory_type == \"%s\"", string(memType))
		}
		typeFilter += ")"
		filterExpr = fmt.Sprintf("%s && %s", filterExpr, typeFilter)
	}

	logging.Debugf("MilvusStore.Retrieve: filter expression: %s", filterExpr)

	// Create search parameters
	// Using HNSW index with ef parameter (adjust based on your index configuration)
	searchParam, err := entity.NewIndexHNSWSearchParam(64)
	if err != nil {
		return nil, fmt.Errorf("Failed to create search parameters: %w", err)
	}

	// Perform vector search in Milvus with retry logic
	// We search for top-k results, then filter by threshold
	searchTopK := limit * 4 // Search for more results to account for threshold filtering
	if searchTopK < 20 {
		searchTopK = 20 // Minimum search size
	}

	var searchResult []client.SearchResult
	err = m.retryWithBackoff(ctx, func() error {
		var retryErr error
		searchResult, retryErr = m.client.Search(
			ctx,
			m.collectionName,
			[]string{}, // Empty partitions means search all
			filterExpr,
			[]string{"id", "content", "memory_type", "metadata"},
			[]entity.Vector{entity.FloatVector(embedding)},
			"embedding", // Vector field name
			entity.COSINE, // Metric type
			searchTopK,
			searchParam,
		)
		return retryErr
	})
	if err != nil {
		return nil, fmt.Errorf("Milvus search failed after retries: %w", err)
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		logging.Debugf("MilvusStore.Retrieve: no results found")
		return []RetrieveResult{}, nil
	}

	// Extract results and filter by threshold
	results := make([]RetrieveResult, 0, limit)
	scores := searchResult[0].Scores
	fields := searchResult[0].Fields

	// Find field indices
	idIdx, contentIdx, typeIdx, metadataIdx := -1, -1, -1, -1
	for i, field := range fields {
		fieldName := field.Name()
		switch fieldName {
		case "id": idIdx = i
		case "content": contentIdx = i
		case "memory_type": typeIdx = i
		case "metadata": metadataIdx = i
		}
		logging.Debugf("MilvusStore.Retrieve: field[%d] name='%s'", i, fieldName)
	}
	logging.Debugf("MilvusStore.Retrieve: field indices - id=%d, content=%d, type=%d, metadata=%d",
		idIdx, contentIdx, typeIdx, metadataIdx)

	// Process results and filter by threshold
	for i := 0; i < len(scores) && len(results) < limit; i++ {
		// Filter by similarity threshold
		score := scores[i]
		if score < threshold {
			logging.Debugf("MilvusStore.Retrieve: skipping result %d with score %.4f < threshold %.4f",
				i, score, threshold)
			continue
		}

		// Extract fields
		result := RetrieveResult{
			Similarity: score,
			Metadata:   make(map[string]interface{}),
		}

		// Extract ID
		if idIdx >= 0 && idIdx < len(fields) {
			if col, ok := fields[idIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if val, err := col.ValueByIdx(i); err == nil {
					result.ID = val
				}
			}
		}

		// Extract content
		if contentIdx >= 0 && contentIdx < len(fields) {
			if col, ok := fields[contentIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if val, err := col.ValueByIdx(i); err == nil {
					result.Content = val
				}
			}
		}

		// Extract memory type
		if typeIdx >= 0 && typeIdx < len(fields) {
			if col, ok := fields[typeIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if val, err := col.ValueByIdx(i); err == nil {
					result.Type = MemoryType(val)
				}
			}
		}

		// Extract metadata (if available as JSON string)
		if metadataIdx >= 0 && metadataIdx < len(fields) {
			if col, ok := fields[metadataIdx].(*entity.ColumnVarChar); ok && col.Len() > i {
				if metadataVal, err := col.ValueByIdx(i); err == nil && metadataVal != "" {
					// Inflate JSON string into the map for downstream code accessibility
					if err := json.Unmarshal([]byte(metadataVal), &result.Metadata); err != nil {
						// Fallback if JSON is malformed
						result.Metadata["raw"] = metadataVal
					} else {
						// Reference for debugging/audit
						result.Metadata["_raw_source"] = metadataVal
					}
				}
			}
		}

		// Only add if we have at least ID and content
		if result.ID != "" && result.Content != "" {
			results = append(results, result)
		}
	}

	logging.Debugf("MilvusStore.Retrieve: returning %d results (filtered from %d candidates)",
		len(results), len(scores))

	return results, nil
}

// IsEnabled returns whether the store is enabled
func (m *MilvusStore) IsEnabled() bool {
	return m.enabled
}

// CheckConnection verifies the Milvus connection is healthy
func (m *MilvusStore) CheckConnection(ctx context.Context) error {
	if !m.enabled {
		return nil
	}

	if m.client == nil {
		return fmt.Errorf("milvus client is not initialized")
	}

	// Check if collection exists
	hasCollection, err := m.client.HasCollection(ctx, m.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	if !hasCollection {
		return fmt.Errorf("collection '%s' does not exist", m.collectionName)
	}

	return nil
}

// Close releases resources held by the store
func (m *MilvusStore) Close() error {
	// Note: We don't close the client here as it might be shared
	// The caller is responsible for managing the client lifecycle
	return nil
}

// isTransientError checks if an error is transient and should be retried
func isTransientError(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())

	// Check for common transient error patterns
	transientPatterns := []string{
		"connection",
		"timeout",
		"deadline exceeded",
		"context deadline exceeded",
		"unavailable",
		"temporary",
		"retry",
		"rate limit",
		"too many requests",
		"server error",
		"internal error",
		"service unavailable",
		"network",
		"broken pipe",
		"connection reset",
		"no connection",
		"connection refused",
	}

	for _, pattern := range transientPatterns {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}

	return false
}

// retryWithBackoff retries an operation with exponential backoff for transient errors
func (m *MilvusStore) retryWithBackoff(ctx context.Context, operation func() error) error {
	var lastErr error

	for attempt := 0; attempt < m.maxRetries; attempt++ {
		lastErr = operation()

		// If no error or non-transient error, return immediately
		if lastErr == nil || !isTransientError(lastErr) {
			return lastErr
		}

		// If this is the last attempt, return the error
		if attempt == m.maxRetries-1 {
			logging.Warnf("MilvusStore: operation failed after %d retries: %v", m.maxRetries, lastErr)
			return lastErr
		}

		// Calculate exponential backoff delay
		delay := m.retryBaseDelay * time.Duration(1<<uint(attempt)) // 2^attempt * baseDelay

		logging.Debugf("MilvusStore: transient error on attempt %d/%d, retrying in %v: %v",
			attempt+1, m.maxRetries, delay, lastErr)

		// Wait with context cancellation support
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled during retry: %w", ctx.Err())
		case <-time.After(delay):
			// Continue to next retry
		}
	}

	return lastErr
}

