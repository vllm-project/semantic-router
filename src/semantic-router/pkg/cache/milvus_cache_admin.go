package cache

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Close releases all resources held by the cache
func (c *MilvusCache) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}

func milvusExtractDocumentRowContent(fields []entity.Column, rowIdx int) (string, bool) {
	for _, field := range fields {
		contentCol, ok := field.(*entity.ColumnVarChar)
		if !ok || contentCol.Len() <= rowIdx {
			continue
		}
		fieldValue, err := contentCol.ValueByIdx(rowIdx)
		if err != nil || fieldValue == "" {
			continue
		}
		if len(fieldValue) == 32 && isHexString(fieldValue) {
			continue
		}
		return fieldValue, true
	}
	return "", false
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
//
//nolint:gocognit,cyclop,funlen,nestif
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
			continue
		}

		content, found := milvusExtractDocumentRowContent(searchResult[0].Fields, i)
		if found && content != "" {
			contents = append(contents, content)
			scores = append(scores, score)
			continue
		}
		logging.Warnf("SearchDocuments: could not extract content for result %d (score=%.3f)", i, score)
	}

	return contents, scores, nil
}

// GetStats provides current cache performance metrics
//
//nolint:nestif
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
