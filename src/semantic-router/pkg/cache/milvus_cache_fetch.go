package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

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

// GetByID retrieves a document from Milvus by its request ID
// This is much more efficient than FindSimilar when you already know the ID
// Used by hybrid cache to fetch documents after local HNSW search
//
//nolint:funlen,cyclop,nestif
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
