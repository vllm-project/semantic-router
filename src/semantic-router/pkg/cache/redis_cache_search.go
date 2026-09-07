package cache

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	valkeyutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/valkey"
)

// recordCacheMiss records a cache miss with the given status and logs the event.
func (c *RedisCache) recordCacheMiss(status string, elapsed time.Duration) {
	atomic.AddInt64(&c.missCount, 1)
	metrics.RecordCacheOperation("redis", "find_similar", status, elapsed.Seconds())
}

// extractSearchResult parses the best match from a search result and returns
// the similarity score, response body, storedAt, and expiresAt. Returns (0, nil, time.Time{}, time.Time{}, false) on failure.
func (c *RedisCache) extractSearchResult(bestDoc redis.Document) (float32, []byte, time.Time, time.Time, bool) {
	distanceVal, ok := bestDoc.Fields["vector_distance"]
	if !ok {
		logging.Infof("RedisCache: vector_distance field not found in result")
		return 0, nil, time.Time{}, time.Time{}, false
	}

	var distance float64
	if _, err := fmt.Sscanf(fmt.Sprint(distanceVal), "%f", &distance); err != nil {
		logging.Infof("RedisCache: failed to parse distance value: %v", err)
		return 0, nil, time.Time{}, time.Time{}, false
	}

	similarity := float32(valkeyutil.DistanceToSimilarity(c.config.Index.VectorField.MetricType, distance))

	responseBodyStr := fmt.Sprint(bestDoc.Fields["response_body"])
	if responseBodyStr == "" {
		logging.Infof("RedisCache: response_body is empty - treating as miss")
		return similarity, nil, time.Time{}, time.Time{}, false
	}

	var storedAt time.Time
	var expiresAt time.Time
	if tsVal, exists := bestDoc.Fields["timestamp"]; exists {
		var ts int64
		if _, err := fmt.Sscanf(fmt.Sprint(tsVal), "%d", &ts); err == nil && ts > 0 {
			storedAt = time.Unix(ts, 0)
		}
	}
	if ttlVal, exists := bestDoc.Fields["ttl_seconds"]; exists && !storedAt.IsZero() {
		var ttlSec int64
		if _, err := fmt.Sscanf(fmt.Sprint(ttlVal), "%d", &ttlSec); err == nil && ttlSec > 0 {
			expiresAt = storedAt.Add(time.Duration(ttlSec) * time.Second)
		}
	}

	return similarity, []byte(responseBodyStr), storedAt, expiresAt, true
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *RedisCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	result, err := c.LookupSimilarWithThreshold(context.Background(), model, query, threshold)
	return result.ResponseBody, result.Found, err
}

func (c *RedisCache) executeFTSearch(ctx context.Context, model string, embeddingBytes []byte) (redis.FTSearchResult, error) {
	knnQuery := partitionedKNNQuery(model, c.config.Search.TopK, c.config.Index.VectorField.Name)
	searchOptions := &redis.FTSearchOptions{
		Return: []redis.FTSearchReturn{
			{FieldName: "vector_distance"},
			{FieldName: "response_body"},
			{FieldName: "timestamp"},
			{FieldName: "ttl_seconds"},
		},
		DialectVersion: 2,
		Params: map[string]interface{}{
			"vec": embeddingBytes,
		},
	}
	if c.searchFn != nil {
		return c.searchFn(ctx, c.indexName, knnQuery, searchOptions)
	}
	return c.client.FTSearchWithArgs(ctx, c.indexName, knnQuery, searchOptions).Result()
}

// LookupSimilarWithThreshold returns response data and similarity atomically.
func (c *RedisCache) LookupSimilarWithThreshold(ctx context.Context, model string, query string, threshold float32) (LookupResult, error) {
	start := time.Now()

	if !c.enabled {
		return LookupResult{}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}

	queryEmbedding, err := c.getEmbedding(ctx, query)
	if err != nil {
		metrics.RecordCacheOperation("redis", "find_similar", "error", time.Since(start).Seconds())
		return LookupResult{}, fmt.Errorf("failed to generate embedding: %w", err)
	}

	embeddingBytes := floatsToBytes(queryEmbedding)
	searchResult, err := c.executeFTSearch(ctx, model, embeddingBytes)
	if err != nil {
		logging.Infof("RedisCache.FindSimilarWithThreshold: search failed: %v", err)
		c.recordCacheMiss("error", time.Since(start))
		if contextErr := contextErrorOnFailure(ctx, err); contextErr != nil {
			return LookupResult{}, contextErr
		}
		return LookupResult{}, nil
	}

	if searchResult.Total == 0 {
		c.recordCacheMiss("miss", time.Since(start))
		return LookupResult{}, nil
	}

	similarity, responseBody, storedAt, expiresAt, ok := c.extractSearchResult(searchResult.Docs[0])
	if !ok {
		c.recordCacheMiss("error", time.Since(start))
		return LookupResult{Similarity: similarity}, nil
	}

	logging.Infof("Similarity=%.4f, threshold=%.4f (metric=%s)",
		similarity, threshold, c.config.Index.VectorField.MetricType)

	if similarity < threshold {
		logging.LogEvent("cache_miss", map[string]interface{}{
			"backend":         "redis",
			"best_similarity": similarity,
			"threshold":       threshold,
			"model":           model,
			"index":           c.indexName,
		})
		c.recordCacheMiss("miss", time.Since(start))
		// The rejected candidate's score belongs to this lookup; see the
		// in-memory backend for the full rationale.
		return LookupResult{Similarity: similarity}, nil
	}

	atomic.AddInt64(&c.hitCount, 1)
	logging.LogEvent("cache_hit", map[string]interface{}{
		"backend":    "redis",
		"similarity": similarity,
		"threshold":  threshold,
		"model":      model,
		"index":      c.indexName,
	})
	metrics.RecordCacheOperation("redis", "find_similar", "hit", time.Since(start).Seconds())
	return lookupResultFromTimestamps(responseBody, similarity, storedAt, expiresAt), nil
}
