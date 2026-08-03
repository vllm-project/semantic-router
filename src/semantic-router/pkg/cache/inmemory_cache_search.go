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
//  2. the exact model partition drops entries owned by another model/recipe;
//  3. the hard user-scope gate (see CacheScopeNamespaceOf) drops a different
//     user's entry BEFORE the expiry check, so an out-of-scope entry never
//     counts toward expiredCount;
//  4. expired entries are not matchable but are reported via expired=true so
//     the caller can tally them.
func (c *InMemoryCache) entryEligible(entry CacheEntry, model, scopeNamespace string, now time.Time) (ok bool, expired bool) {
	if entry.ResponseBody == nil {
		return false, false
	}
	if entry.Model != model {
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

type cacheSearchResult struct {
	bestIndex             int
	bestEntry             CacheEntry
	bestSimilarity        float32
	entriesChecked        int
	expiredCount          int
	polarityRejected      bool
	polarityRejectedEntry CacheEntry
	polarityRejectedScore float32
}

func (c *InMemoryCache) considerSearchCandidate(
	result *cacheSearchResult,
	query string,
	threshold float32,
	entryIndex int,
	entry CacheEntry,
	queryEmbedding []float32,
) {
	dotProduct := embeddingDotProduct(queryEmbedding, entry.Embedding)
	result.entriesChecked++

	if dotProduct >= threshold && polarityMismatch(query, entry.Query) {
		if !result.polarityRejected || dotProduct > result.polarityRejectedScore {
			result.polarityRejected = true
			result.polarityRejectedEntry = entry
			result.polarityRejectedScore = dotProduct
		}
		return
	}

	if result.bestIndex == -1 || dotProduct > result.bestSimilarity {
		result.bestSimilarity = dotProduct
		result.bestIndex = entryIndex
		result.bestEntry = entry
	}
}

func (c *InMemoryCache) scanHNSWCandidates(
	queryEmbedding []float32,
	model string,
	query string,
	threshold float32,
	scopeNamespace string,
	now time.Time,
) cacheSearchResult {
	result := cacheSearchResult{bestIndex: -1}
	candidateIndices := c.hnswIndex.searchKNN(queryEmbedding, 10, c.hnswEfSearch, c.entries)
	for _, entryIndex := range candidateIndices {
		if entryIndex < 0 || entryIndex >= len(c.entries) {
			continue
		}
		entry := c.entries[entryIndex]
		ok, expired := c.entryEligible(entry, model, scopeNamespace, now)
		if expired {
			result.expiredCount++
		}
		if !ok {
			continue
		}
		c.considerSearchCandidate(&result, query, threshold, entryIndex, entry, queryEmbedding)
	}
	logging.Debugf("InMemoryCache.FindSimilar: HNSW search checked %d candidates", len(candidateIndices))
	return result
}

func (c *InMemoryCache) scanLinearForSimilarity(
	queryEmbedding []float32,
	model string,
	query string,
	threshold float32,
	scopeNamespace string,
	now time.Time,
) cacheSearchResult {
	result := cacheSearchResult{bestIndex: -1}
	for entryIndex, entry := range c.entries {
		ok, expired := c.entryEligible(entry, model, scopeNamespace, now)
		if expired {
			result.expiredCount++
		}
		if !ok {
			continue
		}
		c.considerSearchCandidate(&result, query, threshold, entryIndex, entry, queryEmbedding)
	}
	if !c.useHNSW {
		logging.Debugf("InMemoryCache.FindSimilar: Linear search used (HNSW disabled)")
	}
	return result
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
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: searching for model='%s', query=%s, threshold=%.4f",
		model, logging.ContentDescriptor(query), threshold)

	queryEmbedding, err := c.generateEmbedding(query)
	if err != nil {
		metrics.RecordCacheOperation("memory", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	search := c.runFindSimilarEmbeddingSearch(
		queryEmbedding, model, query, threshold, CacheScopeNamespaceOf(query),
	)

	return c.finishFindSimilarSearch(start, model, query, threshold, search)
}

func (c *InMemoryCache) runFindSimilarEmbeddingSearch(
	queryEmbedding []float32,
	model string,
	query string,
	threshold float32,
	scopeNamespace string,
) cacheSearchResult {
	c.mu.RLock()
	now := time.Now()
	var result cacheSearchResult
	if c.useHNSW && c.hnswIndex != nil {
		c.refreshHNSWIfStaleDuringSearch()
		result = c.scanHNSWCandidates(queryEmbedding, model, query, threshold, scopeNamespace, now)
		if result.polarityRejected && (result.bestIndex < 0 || result.bestSimilarity < threshold) {
			// HNSW only considers an approximate candidate set. Once its nearest
			// results are rejected for polarity, scan all entries before treating
			// the lookup as a miss so a valid lower-ranked cache hit is not hidden.
			result = c.scanLinearForSimilarity(queryEmbedding, model, query, threshold, scopeNamespace, now)
		}
	} else {
		result = c.scanLinearForSimilarity(queryEmbedding, model, query, threshold, scopeNamespace, now)
	}
	c.mu.RUnlock()
	return result
}

// recordPolarityReject accounts a polarity-guard rejection as a cache miss and
// emits the observability signal for it. The caller-visible outcome is a miss
// (the request is recomputed); a dedicated cache_negation_reject event lets the
// rejection be distinguished from an ordinary below-threshold miss without
// adding a hot-path metric label.
func (c *InMemoryCache) recordPolarityReject(start time.Time, model, query, cachedQuery string, similarity, threshold float32) {
	atomic.AddInt64(&c.missCount, 1)
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: POLARITY REJECT - similarity=%.4f >= threshold=%.4f but query and cached entry differ in polarity (negation/antonym); treating as miss",
		similarity, threshold)
	logging.LogEvent("cache_negation_reject", map[string]interface{}{
		"backend":      "memory",
		"similarity":   similarity,
		"threshold":    threshold,
		"model":        model,
		"query":        logging.ContentDescriptor(query),
		"cached_query": logging.ContentDescriptor(cachedQuery),
	})
	metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
}

func (c *InMemoryCache) finishFindSimilarSearch(
	start time.Time,
	model string,
	query string,
	threshold float32,
	search cacheSearchResult,
) ([]byte, bool, error) {
	if search.expiredCount > 0 {
		logging.Debugf("InMemoryCache: excluded %d expired entries during search (TTL: %ds)",
			search.expiredCount, c.ttlSeconds)
		logging.LogEvent("cache_expired_entries_found", map[string]interface{}{
			"backend":       "memory",
			"expired_count": search.expiredCount,
			"ttl_seconds":   c.ttlSeconds,
		})
	}

	// If every above-threshold candidate was rejected, preserve the old miss
	// semantics and diagnostics. A valid lower-ranked candidate, however, is
	// selected by the scan and must remain eligible for a cache hit.
	if search.polarityRejected && (search.bestIndex < 0 || search.bestSimilarity < threshold) {
		c.StoreSimilarity(search.polarityRejectedScore)
		c.recordPolarityReject(
			start, model, query, search.polarityRejectedEntry.Query,
			search.polarityRejectedScore, threshold,
		)
		return nil, false, nil
	}

	if search.bestIndex < 0 {
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("InMemoryCache.FindSimilarWithThreshold: no entries found with responses")
		metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
		return nil, false, nil
	}

	c.StoreSimilarity(search.bestSimilarity)

	if search.bestSimilarity >= threshold {
		// The polarity guard (#2691) runs per candidate in considerSearchCandidate:
		// any above-threshold negation/antonym variant is diverted to
		// polarityRejected before it can win bestEntry, and the all-rejected case
		// is handled above. So a winner reaching here is already known not to be a
		// polarity mismatch — no second check is needed on the hot path.
		atomic.AddInt64(&c.hitCount, 1)

		c.mu.Lock()
		c.updateAccessInfo(search.bestIndex, search.bestEntry)
		c.mu.Unlock()

		logging.Debugf("InMemoryCache.FindSimilarWithThreshold: CACHE HIT - similarity=%.4f >= threshold=%.4f, response_size=%d bytes",
			search.bestSimilarity, threshold, len(search.bestEntry.ResponseBody))
		logging.LogEvent("cache_hit", map[string]interface{}{
			"backend":    "memory",
			"similarity": search.bestSimilarity,
			"threshold":  threshold,
			"model":      model,
		})
		metrics.RecordCacheOperation("memory", "find_similar", "hit", time.Since(start).Seconds())
		return search.bestEntry.ResponseBody, true, nil
	}

	atomic.AddInt64(&c.missCount, 1)
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: CACHE MISS - best_similarity=%.4f < threshold=%.4f (checked %d entries)",
		search.bestSimilarity, threshold, search.entriesChecked)
	logging.LogEvent("cache_miss", map[string]interface{}{
		"backend":         "memory",
		"best_similarity": search.bestSimilarity,
		"threshold":       threshold,
		"model":           model,
		"entries_checked": search.entriesChecked,
	})
	metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
	return nil, false, nil
}
