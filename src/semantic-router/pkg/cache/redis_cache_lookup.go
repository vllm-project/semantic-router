package cache

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// FindSimilar searches for semantically similar cached requests.
func (c *RedisCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold.
func (c *RedisCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	start := time.Now()
	logging.Infof("FindSimilarWithThreshold ENTERED: model=%s, query='%s', threshold=%.2f", model, query, threshold)

	if !c.enabled {
		logging.Infof("FindSimilarWithThreshold: cache disabled, returning early")
		return nil, false, nil
	}

	logging.Infof("FindSimilarWithThreshold: cache enabled, generating embedding for query")
	queryEmbedding, err := c.getEmbedding(query)
	if err != nil {
		c.recordFindSimilarMetric(start, "error")
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	searchResult, err := c.searchSimilarDocuments(floatsToBytes(queryEmbedding))
	if err != nil {
		logging.Infof("RedisCache.FindSimilarWithThreshold: search failed: %v", err)
		c.recordFindSimilarFailure(start)
		return nil, false, nil
	}

	logging.Infof("RedisCache.FindSimilarWithThreshold: search returned %d results", searchResult.Total)
	if searchResult.Total == 0 {
		atomic.AddInt64(&c.missCount, 1)
		logging.Infof("RedisCache.FindSimilarWithThreshold: no entries found - cache miss")
		c.recordFindSimilarMetric(start, "miss")
		return nil, false, nil
	}

	bestDoc := searchResult.Docs[0]
	similarity, ok := c.responseSimilarity(bestDoc)
	if !ok {
		c.recordFindSimilarFailure(start)
		return nil, false, nil
	}

	logging.Infof("Calculated similarity=%.4f, threshold=%.4f (metric=%s)",
		similarity, threshold, c.config.Index.VectorField.MetricType)
	if similarity < threshold {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("RedisCache.FindSimilarWithThreshold: cache miss - similarity %.4f below threshold %.4f",
			similarity, threshold)
		logging.LogEvent("cache_miss", map[string]interface{}{
			"backend":         c.backendLabel,
			"best_similarity": similarity,
			"threshold":       threshold,
			"model":           model,
			"index":           c.indexName,
		})
		c.recordFindSimilarMetric(start, "miss")
		return nil, false, nil
	}

	responseBody, ok := c.responseBody(bestDoc)
	if !ok {
		c.recordFindSimilarFailure(start)
		return nil, false, nil
	}

	logging.Infof("CACHE HIT: Found cached response, similarity=%.4f, response_size=%d bytes", similarity, len(responseBody))
	atomic.AddInt64(&c.hitCount, 1)
	logging.Debugf("RedisCache.FindSimilarWithThreshold: cache hit - similarity=%.4f, response_size=%d bytes",
		similarity, len(responseBody))
	logging.LogEvent("cache_hit", map[string]interface{}{
		"backend":    c.backendLabel,
		"similarity": similarity,
		"threshold":  threshold,
		"model":      model,
		"index":      c.indexName,
	})
	c.recordFindSimilarMetric(start, "hit")
	return responseBody, true, nil
}

func (c *RedisCache) searchSimilarDocuments(embeddingBytes []byte) (*redis.FTSearchResult, error) {
	ctx := context.Background()
	knnQuery := fmt.Sprintf("[KNN %d @%s $vec AS vector_distance]",
		c.config.Search.TopK, c.config.Index.VectorField.Name)

	result, err := c.client.FTSearchWithArgs(ctx,
		c.indexName,
		knnQuery,
		&redis.FTSearchOptions{
			Return: []redis.FTSearchReturn{
				{FieldName: "vector_distance"},
				{FieldName: "response_body"},
			},
			DialectVersion: 2,
			Params: map[string]interface{}{
				"vec": embeddingBytes,
			},
		},
	).Result()
	if err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *RedisCache) responseSimilarity(bestDoc redis.Document) (float32, bool) {
	logging.Infof("Extracting fields from best match document...")
	distanceVal, ok := bestDoc.Fields["vector_distance"]
	if !ok {
		logging.Infof("RedisCache.FindSimilarWithThreshold: vector_distance field not found in result")
		return 0, false
	}

	var distance float64
	if _, err := fmt.Sscanf(fmt.Sprint(distanceVal), "%f", &distance); err != nil {
		logging.Infof("RedisCache.FindSimilarWithThreshold: failed to parse distance value: %v", err)
		return 0, false
	}

	return c.distanceToSimilarity(distance), true
}

func (c *RedisCache) distanceToSimilarity(distance float64) float32 {
	switch c.config.Index.VectorField.MetricType {
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

func (c *RedisCache) responseBody(bestDoc redis.Document) ([]byte, bool) {
	logging.Infof("Attempting to extract response_body field...")
	responseBodyVal, ok := bestDoc.Fields["response_body"]
	if !ok {
		logging.Infof("RedisCache.FindSimilarWithThreshold: cache hit BUT response_body field is MISSING - treating as miss")
		return nil, false
	}

	responseBodyStr := fmt.Sprint(responseBodyVal)
	if responseBodyStr == "" {
		logging.Infof("RedisCache.FindSimilarWithThreshold: cache hit BUT response_body is EMPTY - treating as miss")
		return nil, false
	}

	return []byte(responseBodyStr), true
}

func (c *RedisCache) recordFindSimilarFailure(start time.Time) {
	atomic.AddInt64(&c.missCount, 1)
	c.recordFindSimilarMetric(start, "error")
}

func (c *RedisCache) recordFindSimilarMetric(start time.Time, outcome string) {
	metrics.RecordCacheOperation(c.backendLabel, "find_similar", outcome, time.Since(start).Seconds())
}
