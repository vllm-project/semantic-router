package cache

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func milvusActiveEntryFilterExpr() string {
	return fmt.Sprintf(
		`response_body != "" && (expires_at == 0 || expires_at > %d)`,
		time.Now().Unix(),
	)
}

func milvusResponseBodyFieldIndex(hit *client.SearchResult) int {
	if hit == nil || len(hit.Fields) <= 1 {
		return 0
	}
	testCol, ok := hit.Fields[0].(*entity.ColumnVarChar)
	if !ok || testCol.Len() == 0 {
		return 0
	}
	testVal := testCol.Data()[0]
	if len(testVal) == 32 && isHexString(testVal) {
		return 1
	}
	return 0
}

func milvusFirstVarCharBytes(hit *client.SearchResult, fieldIdx int) []byte {
	if hit == nil || fieldIdx < 0 || fieldIdx >= len(hit.Fields) {
		return nil
	}
	col, ok := hit.Fields[fieldIdx].(*entity.ColumnVarChar)
	if !ok || col.Len() == 0 {
		return nil
	}
	return []byte(col.Data()[0])
}

func (c *MilvusCache) milvusSearchSimilarVectors(ctx context.Context, queryEmbedding []float32) ([]client.SearchResult, error) {
	searchParam, err := entity.NewIndexHNSWSearchParam(c.config.Search.Params.Ef)
	if err != nil {
		return nil, err
	}
	return c.client.Search(
		ctx,
		c.collectionName,
		[]string{},
		milvusActiveEntryFilterExpr(),
		[]string{"response_body"},
		[]entity.Vector{entity.FloatVector(queryEmbedding)},
		c.config.Collection.VectorField.Name,
		entity.MetricType(c.config.Collection.VectorField.MetricType),
		c.config.Search.TopK,
		searchParam,
	)
}

// FindSimilar searches for semantically similar cached requests
func (c *MilvusCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
//
//nolint:cyclop,funlen
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

	searchResult, err := c.milvusSearchSimilarVectors(context.Background(), queryEmbedding)
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

	hit := &searchResult[0]
	bestScore := hit.Scores[0]
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

	idx := milvusResponseBodyFieldIndex(hit)
	responseBody := milvusFirstVarCharBytes(hit, idx)
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

// isHexString checks if a string contains only hexadecimal characters
func isHexString(s string) bool {
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') && (c < 'A' || c > 'F') {
			return false
		}
	}
	return true
}
