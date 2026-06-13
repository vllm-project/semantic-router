//go:build !windows && cgo

package cache

import (
	"fmt"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func embeddingDotProduct(queryEmbedding, candidate []float32) float32 {
	var dot float32
	for i := 0; i < len(queryEmbedding) && i < len(candidate); i++ {
		dot += queryEmbedding[i] * candidate[i]
	}
	return dot
}

func (c *InMemoryCache) refreshHNSWIfStaleDuringSearch() {
	if !c.hnswNeedsRebuild {
		return
	}
	logging.Debugf("InMemoryCache.FindSimilar: HNSW index marked as needing rebuild, rebuilding now")
	c.mu.RUnlock()
	c.mu.Lock()
	if c.hnswNeedsRebuild {
		c.rebuildHNSWIndex()
	}
	c.mu.Unlock()
	c.mu.RLock()
}

// entryEligible reports whether an entry may be returned as a search match for
// the requester's scope (ok), and separately whether it was skipped because it
// has expired (expired). The two search paths share it so the per-candidate
// skip logic — and the security-critical scope gate in particular — lives in
// exactly one place that neither path can silently drop.
//
// The check order is load-bearing:
//  1. entries without a stored response are not matchable;
//  2. the hard user-scope gate (see CacheScopeNamespaceOf) drops a different
//     user's entry BEFORE the expiry check, so an out-of-scope entry never
//     counts toward expiredCount;
//  3. expired entries are not matchable but are reported via expired=true so
//     the caller can tally them.
func (c *InMemoryCache) entryEligible(entry CacheEntry, scopeNamespace string, now time.Time) (ok bool, expired bool) {
	if entry.ResponseBody == nil {
		return false, false
	}
	// Hard user-scope gate: never return another user's entry even if its
	// embedding is the nearest neighbor.
	if CacheScopeNamespaceOf(entry.Query) != scopeNamespace {
		return false, false
	}
	if c.isExpired(entry, now) {
		return false, true
	}
	return true, false
}

func (c *InMemoryCache) scanHNSWCandidates(
	queryEmbedding []float32,
	scopeNamespace string,
	now time.Time,
) (bestIndex int, bestSimilarity float32, entriesChecked int, expiredCount int) {
	bestIndex = -1
	candidateIndices := c.hnswIndex.searchKNN(queryEmbedding, 10, c.hnswEfSearch, c.entries)
	for _, entryIndex := range candidateIndices {
		if entryIndex < 0 || entryIndex >= len(c.entries) {
			continue
		}
		entry := c.entries[entryIndex]
		ok, expired := c.entryEligible(entry, scopeNamespace, now)
		if expired {
			expiredCount++
		}
		if !ok {
			continue
		}
		dotProduct := embeddingDotProduct(queryEmbedding, entry.Embedding)
		entriesChecked++
		if bestIndex == -1 || dotProduct > bestSimilarity {
			bestSimilarity = dotProduct
			bestIndex = entryIndex
		}
	}
	logging.Debugf("InMemoryCache.FindSimilar: HNSW search checked %d candidates", len(candidateIndices))
	return bestIndex, bestSimilarity, entriesChecked, expiredCount
}

func (c *InMemoryCache) scanLinearForSimilarity(
	queryEmbedding []float32,
	scopeNamespace string,
	now time.Time,
) (bestIndex int, bestSimilarity float32, entriesChecked int, expiredCount int) {
	bestIndex = -1
	for entryIndex, entry := range c.entries {
		ok, expired := c.entryEligible(entry, scopeNamespace, now)
		if expired {
			expiredCount++
		}
		if !ok {
			continue
		}
		dotProduct := embeddingDotProduct(queryEmbedding, entry.Embedding)
		entriesChecked++
		if bestIndex == -1 || dotProduct > bestSimilarity {
			bestSimilarity = dotProduct
			bestIndex = entryIndex
		}
	}
	if !c.useHNSW {
		logging.Debugf("InMemoryCache.FindSimilar: Linear search used (HNSW disabled)")
	}
	return bestIndex, bestSimilarity, entriesChecked, expiredCount
}

// FindSimilar searches for semantically similar cached requests using the default threshold
func (c *InMemoryCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *InMemoryCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	start := time.Now()

	if !c.enabled {
		logging.Debugf("InMemoryCache.FindSimilarWithThreshold: cache disabled")
		return nil, false, nil
	}
	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: searching for model='%s', query='%s' (len=%d chars), threshold=%.4f",
		model, queryPreview, len(query), threshold)

	queryEmbedding, err := c.generateEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("memory", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	bestIndex, bestEntry, bestSimilarity, entriesChecked, expiredCount := c.runFindSimilarEmbeddingSearch(queryEmbedding, CacheScopeNamespaceOf(query))

	return c.finishFindSimilarSearch(
		start, model, threshold,
		bestIndex, bestEntry, bestSimilarity, entriesChecked, expiredCount,
	)
}

func (c *InMemoryCache) runFindSimilarEmbeddingSearch(queryEmbedding []float32, scopeNamespace string) (
	bestIndex int,
	bestEntry CacheEntry,
	bestSimilarity float32,
	entriesChecked int,
	expiredCount int,
) {
	c.mu.RLock()
	now := time.Now()
	if c.useHNSW && c.hnswIndex != nil {
		c.refreshHNSWIfStaleDuringSearch()
		bestIndex, bestSimilarity, entriesChecked, expiredCount = c.scanHNSWCandidates(queryEmbedding, scopeNamespace, now)
	} else {
		bestIndex, bestSimilarity, entriesChecked, expiredCount = c.scanLinearForSimilarity(queryEmbedding, scopeNamespace, now)
	}
	if bestIndex >= 0 {
		bestEntry = c.entries[bestIndex]
	}
	c.mu.RUnlock()
	return bestIndex, bestEntry, bestSimilarity, entriesChecked, expiredCount
}

func (c *InMemoryCache) finishFindSimilarSearch(
	start time.Time,
	model string,
	threshold float32,
	bestIndex int,
	bestEntry CacheEntry,
	bestSimilarity float32,
	entriesChecked int,
	expiredCount int,
) ([]byte, bool, error) {
	if expiredCount > 0 {
		logging.Debugf("InMemoryCache: excluded %d expired entries during search (TTL: %ds)",
			expiredCount, c.ttlSeconds)
		logging.LogEvent("cache_expired_entries_found", map[string]interface{}{
			"backend":       "memory",
			"expired_count": expiredCount,
			"ttl_seconds":   c.ttlSeconds,
		})
	}

	if bestIndex < 0 {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("InMemoryCache.FindSimilarWithThreshold: no entries found with responses")
		metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
		return nil, false, nil
	}

	c.StoreSimilarity(bestSimilarity)

	if bestSimilarity >= threshold {
		atomic.AddInt64(&c.hitCount, 1)

		c.mu.Lock()
		c.updateAccessInfo(bestIndex, bestEntry)
		c.mu.Unlock()

		logging.Debugf("InMemoryCache.FindSimilarWithThreshold: CACHE HIT - similarity=%.4f >= threshold=%.4f, response_size=%d bytes",
			bestSimilarity, threshold, len(bestEntry.ResponseBody))
		logging.LogEvent("cache_hit", map[string]interface{}{
			"backend":    "memory",
			"similarity": bestSimilarity,
			"threshold":  threshold,
			"model":      model,
		})
		metrics.RecordCacheOperation("memory", "find_similar", "hit", time.Since(start).Seconds())
		return bestEntry.ResponseBody, true, nil
	}

	atomic.AddInt64(&c.missCount, 1)
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: CACHE MISS - best_similarity=%.4f < threshold=%.4f (checked %d entries)",
		bestSimilarity, threshold, entriesChecked)
	logging.LogEvent("cache_miss", map[string]interface{}{
		"backend":         "memory",
		"best_similarity": bestSimilarity,
		"threshold":       threshold,
		"model":           model,
		"entries_checked": entriesChecked,
	})
	metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
	return nil, false, nil
}
