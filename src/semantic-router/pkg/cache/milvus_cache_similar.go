package cache

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func milvusStringLiteral(value string) string {
	escaped := strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(value)
	return `"` + escaped + `"`
}

func milvusActiveEntryFilterExpr(model string) string {
	return fmt.Sprintf(
		`model == %s && query != %s && response_body != "" && (expires_at == 0 || expires_at > %d)`,
		milvusStringLiteral(model),
		milvusStringLiteral(exactCacheQueryMarker),
		time.Now().Unix(),
	)
}

// milvusEntryAt decodes only named fields; primary-key order and response
// contents are not a reliable way to identify a stored query or response.
func milvusEntryAt(fields client.ResultSet, index int) CacheEntry {
	var entry CacheEntry
	for _, field := range fields {
		switch col := field.(type) {
		case *entity.ColumnVarChar:
			if index >= col.Len() {
				continue
			}
			value, err := col.ValueByIdx(index)
			if err != nil {
				continue
			}
			switch col.Name() {
			case "query":
				entry.Query = value
			case "response_body":
				entry.ResponseBody = []byte(value)
			}
		case *entity.ColumnInt64:
			if index >= col.Len() {
				continue
			}
			value, err := col.ValueByIdx(index)
			if err != nil || value <= 0 {
				continue
			}
			switch col.Name() {
			case "timestamp":
				entry.Timestamp = time.Unix(value, 0)
			case "expires_at":
				entry.ExpiresAt = time.Unix(value, 0)
			}
		}
	}
	return entry
}

func (c *MilvusCache) milvusSearchSimilarVectors(
	ctx context.Context,
	model string,
	queryEmbedding []float32,
) ([]client.SearchResult, error) {
	if c.searchFn != nil {
		return c.searchFn(ctx, model, queryEmbedding)
	}
	searchParam, err := entity.NewIndexHNSWSearchParam(c.config.Search.Params.Ef)
	if err != nil {
		return nil, err
	}
	return c.client.Search(
		ctx,
		c.collectionName,
		[]string{},
		milvusActiveEntryFilterExpr(model),
		[]string{"query", "response_body", "timestamp", "expires_at"},
		[]entity.Vector{entity.FloatVector(queryEmbedding)},
		c.config.Collection.VectorField.Name,
		entity.MetricType(c.config.Collection.VectorField.MetricType),
		c.config.Search.TopK,
		searchParam,
		c.searchQueryOptions()...,
	)
}

// FindSimilar searches for semantically similar cached requests
func (c *MilvusCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *MilvusCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	result, err := c.LookupSimilarWithThreshold(context.Background(), model, query, threshold)
	return result.ResponseBody, result.Found, err
}

// LookupSimilarWithThreshold returns response data and similarity atomically.
//
//nolint:cyclop,funlen
func (c *MilvusCache) LookupSimilarWithThreshold(ctx context.Context, model string, query string, threshold float32) (LookupResult, error) {
	start := time.Now()

	if !c.enabled {
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: cache disabled")
		return LookupResult{}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	logging.Debugf("MilvusCache.FindSimilarWithThreshold: searching for model='%s', query=%s, threshold=%.4f",
		model, logging.ContentDescriptor(query), threshold)

	queryEmbedding, err := c.getEmbedding(ctx, query)
	if err != nil {
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		return LookupResult{}, fmt.Errorf("failed to generate embedding: %w", err)
	}

	searchResult, err := c.milvusSearchSimilarVectors(ctx, model, queryEmbedding)
	if err != nil {
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: search failed: %v", err)
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		// A canceled or expired request is surfaced as an error; every other
		// search failure keeps the existing fail-open miss (#2473).
		if contextErr := contextErrorOnFailure(ctx, err); contextErr != nil {
			return LookupResult{}, contextErr
		}
		return LookupResult{}, nil
	}

	// Err can coexist with ResultCount == 0, so check it first.
	if len(searchResult) > 0 && searchResult[0].Err != nil {
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: search result error: %v", searchResult[0].Err)
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		return LookupResult{}, nil
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("MilvusCache.FindSimilarWithThreshold: no entries found")
		metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
		return LookupResult{}, nil
	}

	hit := &searchResult[0]
	metricType := c.config.Collection.VectorField.MetricType
	var queryBuffer [64]string
	queryTokens := tokenizeForPolarity(query, queryBuffer[:0])
	bestSimilarity := float32(0)
	haveScore := false
	var selected CacheEntry
	var selectedSimilarity float32
	for index, score := range hit.Scores {
		if index >= hit.ResultCount {
			break
		}
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			continue
		}
		similarity := milvusScoreToSimilarity(metricType, score)
		if math.IsNaN(float64(similarity)) || math.IsInf(float64(similarity), 0) {
			continue
		}
		if !haveScore || similarity > bestSimilarity {
			bestSimilarity, haveScore = similarity, true
		}
		if similarity < threshold {
			continue
		}
		entry := milvusEntryAt(hit.Fields, index)
		if len(entry.ResponseBody) == 0 || !semanticCandidateMatchesPolarity(queryTokens, entry.Query) {
			continue
		}
		if len(selected.ResponseBody) == 0 || similarity > selectedSimilarity {
			selected, selectedSimilarity = entry, similarity
		}
	}
	if len(selected.ResponseBody) > 0 {
		atomic.AddInt64(&c.hitCount, 1)
		logging.LogEvent("cache_hit", map[string]interface{}{
			"backend": "milvus", "similarity": selectedSimilarity, "threshold": threshold,
			"metric": metricType, "model": model, "collection": c.collectionName,
		})
		metrics.RecordCacheOperation("milvus", "find_similar", "hit", time.Since(start).Seconds())
		return lookupResultFromTimestamps(selected.ResponseBody, selectedSimilarity, selected.Timestamp, selected.ExpiresAt), nil
	}
	logging.LogEvent("cache_miss", map[string]interface{}{
		"backend": "milvus", "best_similarity": bestSimilarity, "threshold": threshold,
		"metric": metricType, "model": model, "collection": c.collectionName,
	})
	atomic.AddInt64(&c.missCount, 1)
	metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
	return LookupResult{Similarity: bestSimilarity}, nil
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
