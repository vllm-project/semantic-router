//go:build !windows

package cache

import (
	"context"
	"fmt"
	"sort"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// inMemoryHNSWCandidates is the least number of neighbours an HNSW search
// asks for before the lexical guard and the bounds filter them.
const inMemoryHNSWCandidates = 10

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
	logging.Debugf("InMemoryCache.LookupSimilar: HNSW index marked as needing rebuild, rebuilding now")
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

// inMemoryHit is a search hit and the entry index its access is recorded at.
type inMemoryHit struct {
	index      int
	entry      CacheEntry
	similarity float32
}

type cacheSearchResult struct {
	hits                  []inMemoryHit
	closest               float32
	seen                  bool
	entriesChecked        int
	expiredCount          int
	polarityRejected      bool
	polarityRejectedEntry CacheEntry
	polarityRejectedScore float32
}

func (c *InMemoryCache) considerSearchCandidate(
	result *cacheSearchResult,
	queryTokens []string,
	threshold float32,
	entryIndex int,
	entry CacheEntry,
	queryEmbedding []float32,
) {
	dotProduct := embeddingDotProduct(queryEmbedding, entry.Embedding)
	result.entriesChecked++
	if !result.seen || dotProduct > result.closest {
		result.closest, result.seen = dotProduct, true
	}
	if dotProduct < threshold {
		return
	}

	var cachedBuffer [32]string
	cachedTokens := entry.polarityTokens
	if cachedTokens == nil {
		// Directly constructed entries (including small fixtures) still obey the
		// same guard; production insertion prepares this metadata once.
		cachedTokens = tokenizeForPolarity(entry.Query, cachedBuffer[:0])
	}
	if polarityTokensMismatch(queryTokens, cachedTokens) {
		if !result.polarityRejected || dotProduct > result.polarityRejectedScore {
			result.polarityRejected = true
			result.polarityRejectedEntry = entry
			result.polarityRejectedScore = dotProduct
		}
		return
	}
	result.hits = append(result.hits, inMemoryHit{index: entryIndex, entry: entry, similarity: dotProduct})
}

func (c *InMemoryCache) scanHNSWCandidates(
	queryEmbedding []float32,
	model string,
	queryTokens []string,
	bounds SimilarityBounds,
	scopeNamespace string,
	now time.Time,
) cacheSearchResult {
	result := cacheSearchResult{}
	candidateIndices := c.hnswIndex.searchKNN(queryEmbedding, bounds.fetch(inMemoryHNSWCandidates), c.hnswEfSearch, c.entries)
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
		c.considerSearchCandidate(&result, queryTokens, bounds.MinSimilarity, entryIndex, entry, queryEmbedding)
	}
	logging.Debugf("InMemoryCache.LookupSimilar: HNSW search checked %d candidates", len(candidateIndices))
	return result
}

func (c *InMemoryCache) scanLinearForSimilarity(
	queryEmbedding []float32,
	model string,
	queryTokens []string,
	threshold float32,
	scopeNamespace string,
	now time.Time,
) cacheSearchResult {
	result := cacheSearchResult{}
	for entryIndex, entry := range c.entries {
		ok, expired := c.entryEligible(entry, model, scopeNamespace, now)
		if expired {
			result.expiredCount++
		}
		if !ok {
			continue
		}
		c.considerSearchCandidate(&result, queryTokens, threshold, entryIndex, entry, queryEmbedding)
	}
	if !c.useHNSW {
		logging.Debugf("InMemoryCache.LookupSimilar: Linear search used (HNSW disabled)")
	}
	return result
}

// FindSimilar searches for semantically similar cached requests using the default threshold
func (c *InMemoryCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	return c.FindSimilarWithThreshold(model, query, c.similarityThreshold)
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
func (c *InMemoryCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	result, err := c.LookupSimilarWithThreshold(context.Background(), model, query, threshold)
	return result.ResponseBody, result.Found, err
}

// LookupSimilarWithThreshold returns the most similar entry within threshold.
func (c *InMemoryCache) LookupSimilarWithThreshold(ctx context.Context, model string, query string, threshold float32) (LookupResult, error) {
	return bestSimilar(c.LookupSimilar(ctx, model, query, SimilarityBounds{MinSimilarity: threshold, TopK: 1}))
}

// LookupSimilar returns the request-scoped entries most similar to query.
func (c *InMemoryCache) LookupSimilar(ctx context.Context, model string, query string, bounds SimilarityBounds) (SimilarEntries, error) {
	start := time.Now()

	if !c.enabled {
		logging.Debugf("InMemoryCache.LookupSimilar: cache disabled")
		return SimilarEntries{}, nil
	}
	logging.Debugf("InMemoryCache.LookupSimilar: searching for model='%s', query=%s, threshold=%.4f, top_k=%d",
		model, logging.ContentDescriptor(query), bounds.MinSimilarity, bounds.TopK)

	queryEmbedding, err := c.generateEmbedding(ctx, query)
	if err != nil {
		metrics.RecordCacheOperation("memory", "find_similar", "error", time.Since(start).Seconds())
		return SimilarEntries{}, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Do not return a result if cancellation occurred during embedding.
	if err := ctxErr(ctx); err != nil {
		metrics.RecordCacheOperation("memory", "find_similar", "canceled", time.Since(start).Seconds())
		return SimilarEntries{}, err
	}

	search := c.runFindSimilarEmbeddingSearch(queryEmbedding, model, query, bounds, CacheScopeNamespaceOf(query))
	return c.finishFindSimilarSearch(start, model, query, bounds, search), nil
}

func (c *InMemoryCache) runFindSimilarEmbeddingSearch(
	queryEmbedding []float32,
	model string,
	query string,
	bounds SimilarityBounds,
	scopeNamespace string,
) cacheSearchResult {
	var queryBuffer [32]string
	queryTokens := tokenizeForPolarity(query, queryBuffer[:0])
	c.mu.RLock()
	now := time.Now()
	var result cacheSearchResult
	if c.useHNSW && c.hnswIndex != nil {
		c.refreshHNSWIfStaleDuringSearch()
		result = c.scanHNSWCandidates(queryEmbedding, model, queryTokens, bounds, scopeNamespace, now)
		if result.polarityRejected && len(result.hits) < max(bounds.TopK, 1) {
			// HNSW is approximate; scan all entries before a polarity rejection
			// hides a valid lower-ranked hit.
			result = c.scanLinearForSimilarity(queryEmbedding, model, queryTokens, bounds.MinSimilarity, scopeNamespace, now)
		}
	} else {
		result = c.scanLinearForSimilarity(queryEmbedding, model, queryTokens, bounds.MinSimilarity, scopeNamespace, now)
	}
	c.mu.RUnlock()
	return result
}

// recordLexicalPolarityReject preserves caller-visible miss semantics while emitting
// an event that distinguishes lexical polarity rejection from a threshold miss.
func (c *InMemoryCache) recordLexicalPolarityReject(start time.Time, model, query, cachedQuery string, similarity, threshold float32) {
	atomic.AddInt64(&c.missCount, 1)
	logging.Debugf("InMemoryCache.LookupSimilar: POLARITY REJECT - similarity=%.4f >= threshold=%.4f but query and cached entry differ in polarity (negation/antonym); treating as miss",
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
	bounds SimilarityBounds,
	search cacheSearchResult,
) SimilarEntries {
	if search.expiredCount > 0 {
		logging.Debugf("InMemoryCache: excluded %d expired entries during search (TTL: %ds)",
			search.expiredCount, c.ttlSeconds)
		logging.LogEvent("cache_expired_entries_found", map[string]interface{}{
			"backend":       "memory",
			"expired_count": search.expiredCount,
			"ttl_seconds":   c.ttlSeconds,
		})
	}

	hits := search.hits
	sort.SliceStable(hits, func(i, j int) bool { return hits[i].similarity > hits[j].similarity })
	if bounds.TopK > 0 && len(hits) > bounds.TopK {
		hits = hits[:bounds.TopK]
	}
	entries := SimilarEntries{Closest: search.closest}

	if len(hits) == 0 {
		// A valid lower-ranked candidate remains eligible after stronger
		// candidates are rejected; only a lookup left with none is a miss.
		if search.polarityRejected {
			c.recordLexicalPolarityReject(start, model, query, search.polarityRejectedEntry.Query,
				search.polarityRejectedScore, bounds.MinSimilarity)
			return entries
		}
		atomic.AddInt64(&c.missCount, 1)
		logging.Debugf("InMemoryCache.LookupSimilar: CACHE MISS - closest_similarity=%.4f < threshold=%.4f (checked %d entries)",
			search.closest, bounds.MinSimilarity, search.entriesChecked)
		logging.LogEvent("cache_miss", map[string]interface{}{
			"backend":         "memory",
			"best_similarity": search.closest,
			"threshold":       bounds.MinSimilarity,
			"model":           model,
			"entries_checked": search.entriesChecked,
		})
		metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
		// A rejected candidate's score remains request-owned and is exposed on
		// the debug and Replay surfaces to diagnose near-threshold misses.
		return entries
	}

	atomic.AddInt64(&c.hitCount, 1)
	var queryBuffer [32]string
	queryTokens := tokenizeForPolarity(query, queryBuffer[:0])
	c.mu.Lock()
	for _, hit := range hits {
		c.updateAccessInfo(hit.index, hit.entry)
		result := lookupResultFromTimestamps(hit.entry.ResponseBody, hit.similarity, hit.entry.Timestamp, hit.entry.ExpiresAt)
		result.Query = hit.entry.Query
		result.NegationGuard = negationGuardOutcomeFor(queryTokens, hit.entry.Query)
		entries.Hits = append(entries.Hits, result)
	}
	c.mu.Unlock()
	logging.Debugf("InMemoryCache.LookupSimilar: CACHE HIT - %d entries, best similarity=%.4f >= threshold=%.4f",
		len(entries.Hits), entries.Hits[0].Similarity, bounds.MinSimilarity)
	logging.LogEvent("cache_hit", map[string]interface{}{
		"backend":    "memory",
		"similarity": entries.Hits[0].Similarity,
		"threshold":  bounds.MinSimilarity,
		"model":      model,
	})
	metrics.RecordCacheOperation("memory", "find_similar", "hit", time.Since(start).Seconds())
	return entries
}
