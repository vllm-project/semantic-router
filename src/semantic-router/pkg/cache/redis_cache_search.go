package cache

import (
	"context"
	"fmt"
	"math"
	"strconv"
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
	similarity, ok := c.searchResultSimilarity(bestDoc)
	if !ok {
		return 0, nil, time.Time{}, time.Time{}, false
	}
	body, storedAt, expiresAt, ok := redisSearchPayload(bestDoc)
	return similarity, body, storedAt, expiresAt, ok
}

func (c *RedisCache) searchResultSimilarity(bestDoc redis.Document) (float32, bool) {
	distanceVal, ok := bestDoc.Fields["vector_distance"]
	if !ok {
		logging.Infof("RedisCache: vector_distance field not found in result")
		return 0, false
	}

	distance, err := strconv.ParseFloat(distanceVal, 64)
	if err != nil || math.IsNaN(distance) || math.IsInf(distance, 0) {
		logging.Infof("RedisCache: failed to parse distance value: %v", err)
		return 0, false
	}

	similarity := float32(valkeyutil.DistanceToSimilarity(c.config.Index.VectorField.MetricType, distance))
	return similarity, true
}

func redisSearchPayload(bestDoc redis.Document) ([]byte, time.Time, time.Time, bool) {
	responseBodyStr := bestDoc.Fields["response_body"]
	if responseBodyStr == "" {
		logging.Infof("RedisCache: response_body is empty - treating as miss")
		return nil, time.Time{}, time.Time{}, false
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

	return []byte(responseBodyStr), storedAt, expiresAt, true
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
			{FieldName: "query"},
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

	if err := ctxErr(ctx); err != nil {
		return LookupResult{}, err
	}
	if searchResult.Total == 0 || len(searchResult.Docs) == 0 {
		c.recordCacheMiss("miss", time.Since(start))
		return LookupResult{}, nil
	}

	var queryBuffer [32]string
	queryTokens := tokenizeForPolarity(query, queryBuffer[:0])
	result := c.selectPolarityCandidate(searchResult.Docs, queryTokens, threshold)
	if !result.Found {
		logging.LogEvent("cache_miss", map[string]interface{}{
			"backend":         "redis",
			"best_similarity": result.Similarity,
			"threshold":       threshold,
			"model":           model,
			"index":           c.indexName,
		})
		c.recordCacheMiss("miss", time.Since(start))
		// The rejected candidate's score belongs to this lookup; see the
		// in-memory backend for the full rationale.
		return result, nil
	}

	atomic.AddInt64(&c.hitCount, 1)
	logging.LogEvent("cache_hit", map[string]interface{}{
		"backend":    "redis",
		"similarity": result.Similarity,
		"threshold":  threshold,
		"model":      model,
		"index":      c.indexName,
	})
	metrics.RecordCacheOperation("redis", "find_similar", "hit", time.Since(start).Seconds())
	return result, nil
}

// selectPolarityCandidate considers every fetched candidate before publishing a
// hit, so a rejected nearest neighbor cannot hide a valid lower-ranked result.
func (c *RedisCache) selectPolarityCandidate(docs []redis.Document, queryTokens []string, threshold float32) LookupResult {
	var best LookupResult
	var rejectedSimilarity float32
	hasScore := false
	for _, doc := range docs {
		similarity, validScore := c.searchResultSimilarity(doc)
		if !validScore {
			continue
		}
		if !hasScore || similarity > rejectedSimilarity {
			rejectedSimilarity = similarity
			hasScore = true
		}
		body, storedAt, expiresAt, ok := redisSearchPayload(doc)
		if !ok || similarity < threshold ||
			!semanticCandidateMatchesPolarity(queryTokens, doc.Fields["query"]) {
			continue
		}
		if !expiresAt.IsZero() && !time.Now().Before(expiresAt) {
			continue
		}
		if !best.Found || similarity > best.Similarity {
			best = lookupResultFromTimestamps(body, similarity, storedAt, expiresAt)
		}
	}
	if !best.Found {
		best.Similarity = rejectedSimilarity
	}
	return best
}
