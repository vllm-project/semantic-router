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
	topK int,
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
		topK,
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

// LookupSimilarWithThreshold returns the most similar entry within threshold.
func (c *MilvusCache) LookupSimilarWithThreshold(ctx context.Context, model string, query string, threshold float32) (LookupResult, error) {
	return bestSimilar(c.LookupSimilar(ctx, model, query, SimilarityBounds{MinSimilarity: threshold, TopK: 1}))
}

// LookupSimilar returns the entries most similar to query, with their
// responses, from one search.
//
//nolint:cyclop,funlen
func (c *MilvusCache) LookupSimilar(ctx context.Context, model string, query string, bounds SimilarityBounds) (SimilarEntries, error) {
	start := time.Now()

	if !c.enabled {
		logging.Debugf("MilvusCache.LookupSimilar: cache disabled")
		return SimilarEntries{}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	logging.Debugf("MilvusCache.LookupSimilar: searching for model='%s', query=%s, threshold=%.4f, top_k=%d",
		model, logging.ContentDescriptor(query), bounds.MinSimilarity, bounds.TopK)

	queryEmbedding, err := c.getEmbedding(ctx, query)
	if err != nil {
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		return SimilarEntries{}, fmt.Errorf("failed to generate embedding: %w", err)
	}

	searchResult, err := c.milvusSearchSimilarVectors(ctx, model, queryEmbedding, bounds.fetch(c.config.Search.TopK))
	if err != nil {
		logging.Debugf("MilvusCache.LookupSimilar: search failed: %v", err)
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		// A canceled or expired request is surfaced as an error; every other
		// search failure keeps the existing fail-open miss (#2473).
		if contextErr := contextErrorOnFailure(ctx, err); contextErr != nil {
			return SimilarEntries{}, contextErr
		}
		return SimilarEntries{}, nil
	}

	// Err can coexist with ResultCount == 0, so check it first.
	if len(searchResult) > 0 && searchResult[0].Err != nil {
		logging.Debugf("MilvusCache.LookupSimilar: search result error: %v", searchResult[0].Err)
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("milvus", "find_similar", "error", time.Since(start).Seconds())
		return SimilarEntries{}, nil
	}

	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("MilvusCache.LookupSimilar: no entries found")
		metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
		return SimilarEntries{}, nil
	}

	hit := &searchResult[0]
	metricType := c.config.Collection.VectorField.MetricType
	var queryBuffer [64]string
	queryTokens := tokenizeForPolarity(query, queryBuffer[:0])
	ranked := similarHits{bounds: bounds}
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
		ranked.saw(similarity)
		if similarity < bounds.MinSimilarity {
			continue
		}
		entry := milvusEntryAt(hit.Fields, index)
		if len(entry.ResponseBody) == 0 || !semanticCandidateMatchesPolarity(queryTokens, entry.Query) {
			continue
		}
		result := lookupResultFromTimestamps(entry.ResponseBody, similarity, entry.Timestamp, entry.ExpiresAt)
		result.Query = entry.Query
		result.NegationGuard = negationGuardOutcomeFor(queryTokens, entry.Query)
		ranked.add(result)
	}
	entries := ranked.entries()
	if len(entries.Hits) > 0 {
		atomic.AddInt64(&c.hitCount, 1)
		logging.LogEvent("cache_hit", map[string]interface{}{
			"backend": "milvus", "similarity": entries.Hits[0].Similarity, "threshold": bounds.MinSimilarity,
			"metric": metricType, "model": model, "collection": c.collectionName,
		})
		metrics.RecordCacheOperation("milvus", "find_similar", "hit", time.Since(start).Seconds())
		return entries, nil
	}
	logging.LogEvent("cache_miss", map[string]interface{}{
		"backend": "milvus", "best_similarity": entries.Closest, "threshold": bounds.MinSimilarity,
		"metric": metricType, "model": model, "collection": c.collectionName,
	})
	atomic.AddInt64(&c.missCount, 1)
	metrics.RecordCacheOperation("milvus", "find_similar", "miss", time.Since(start).Seconds())
	return entries, nil
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
