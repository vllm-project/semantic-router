//go:build !windows

package cache

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type candidateWithID struct {
	milvusID   string
	similarity float32
	index      int
}

// FindSimilar searches for semantically similar cached requests.
func (h *HybridCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	result, err := h.LookupSimilarWithThreshold(context.Background(), model, query, h.similarityThreshold)
	return result.ResponseBody, result.Found, err
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold.
func (h *HybridCache) FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error) {
	result, err := h.LookupSimilarWithThreshold(context.Background(), model, query, threshold)
	return result.ResponseBody, result.Found, err
}

// LookupSimilarWithThreshold returns the most similar entry within threshold.
func (h *HybridCache) LookupSimilarWithThreshold(ctx context.Context, model string, query string, threshold float32) (LookupResult, error) {
	return bestSimilar(h.LookupSimilar(ctx, model, query, SimilarityBounds{MinSimilarity: threshold, TopK: 1}))
}

// LookupSimilar returns the entries most similar to query, with their
// responses: from the HNSW candidates when they fill TopK, else from Milvus,
// which holds every entry.
func (h *HybridCache) LookupSimilar(ctx context.Context, model string, query string, bounds SimilarityBounds) (SimilarEntries, error) {
	return h.findSimilar(ctx, model, query, bounds, "find_similar_threshold", "HybridCache.LookupSimilar")
}

func (h *HybridCache) findSimilar(
	ctx context.Context,
	model string,
	query string,
	bounds SimilarityBounds,
	metricOp string,
	logPrefix string,
) (SimilarEntries, error) {
	start := time.Now()
	threshold := bounds.MinSimilarity

	if !h.enabled {
		return SimilarEntries{}, nil
	}

	logging.Debugf("%s: searching for model='%s', query=%s, threshold=%.3f, top_k=%d",
		logPrefix, model, logging.ContentDescriptor(query), threshold, bounds.TopK)

	queryEmbedding, err := h.generateEmbedding(ctx, query)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", metricOp, "error", time.Since(start).Seconds())
		return SimilarEntries{}, fmt.Errorf("failed to generate embedding: %w", err)
	}

	candidatesWithIDs, totalCandidates := h.searchCandidates(queryEmbedding, threshold)
	if len(candidatesWithIDs) == 0 {
		h.recordLookupMiss(start, metricOp, logPrefix, threshold, model, totalCandidates, 0)
		return SimilarEntries{}, nil
	}

	logging.Debugf("%s: HNSW returned %d candidates, %d above threshold",
		logPrefix, totalCandidates, len(candidatesWithIDs))

	var queryTokenBuffer [32]string
	queryTokens := tokenizeForPolarity(query, queryTokenBuffer[:0])
	fetched, fetchErr := h.fetchResponsesFromCandidates(ctx, logPrefix, model, queryTokens, candidatesWithIDs, bounds)
	if len(fetched.hits) >= max(bounds.TopK, 1) {
		entries := fetched.entries()
		h.recordLookupHit(start, metricOp, threshold, model, entries.Hits[0].Similarity)
		return entries, nil
	}

	// The global HNSW graph may return a nearest neighbor from another model
	// partition or one rejected by the lexical polarity floor, and holds only
	// the recent entries. Milvus owns the exact model filter, every entry, and
	// checks each fetched candidate's original query.
	stored, err := h.milvusCache.LookupSimilar(ctx, model, query, bounds)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", metricOp, "error", time.Since(start).Seconds())
		return SimilarEntries{}, err
	}
	if len(stored.Hits) > 0 {
		h.recordLookupHit(start, metricOp, threshold, model, stored.Hits[0].Similarity)
		return stored, nil
	}

	h.recordLookupMiss(start, metricOp, logPrefix, threshold, model, totalCandidates, len(candidatesWithIDs))
	// A missing cross-model candidate is a miss; any other fetch failure is an
	// error and must not be hidden by the cache fail-open path.
	if fetchErr != nil {
		return SimilarEntries{}, fmt.Errorf("%s: candidate fetch failed: %w", logPrefix, fetchErr)
	}
	closest := stored.Closest
	if fetched.seen && fetched.closest > closest {
		closest = fetched.closest
	}
	return SimilarEntries{Closest: closest}, nil
}

func (h *HybridCache) searchCandidates(queryEmbedding []float32, threshold float32) ([]candidateWithID, int) {
	h.mu.RLock()
	candidates := h.searchKNNHybridWithThreshold(queryEmbedding, 1, 20, threshold)
	candidatesWithIDs := h.candidatesAboveThresholdLocked(candidates, threshold)
	h.mu.RUnlock()

	return candidatesWithIDs, len(candidates)
}

func (h *HybridCache) candidatesAboveThresholdLocked(candidates []searchResult, threshold float32) []candidateWithID {
	candidatesWithIDs := make([]candidateWithID, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate.similarity < threshold {
			continue
		}
		if milvusID, ok := h.idMap[candidate.index]; ok {
			candidatesWithIDs = append(candidatesWithIDs, candidateWithID{
				milvusID:   milvusID,
				similarity: candidate.similarity,
				index:      candidate.index,
			})
		}
	}

	return candidatesWithIDs
}

// fetchResponsesFromCandidates fetches the HNSW candidates' entries from
// Milvus, best first, until TopK pass the lexical guard.
func (h *HybridCache) fetchResponsesFromCandidates(
	parent context.Context,
	logPrefix string,
	model string,
	queryTokens []string,
	candidates []candidateWithID,
	bounds SimilarityBounds,
) (similarHits, error) {
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()

	var fetchErr error
	ranked := similarHits{bounds: bounds}
	for _, candidate := range candidates {
		if len(ranked.hits) >= max(bounds.TopK, 1) {
			break
		}
		fetchCtx, fetchCancel := context.WithTimeout(ctx, 2*time.Second)
		entry, err := h.milvusCache.getEntryByID(fetchCtx, candidate.milvusID, model)
		fetchCancel()
		if err != nil {
			if errors.Is(err, errMilvusCacheEntryNotFound) {
				continue
			}
			logging.Debugf("%s: Milvus GetByID failed for %s: %v", logPrefix, candidate.milvusID, err)
			fetchErr = errors.Join(fetchErr, err)
			continue
		}
		// Only an entry of the request's own partition counts as seen.
		ranked.saw(candidate.similarity)
		if len(entry.ResponseBody) > 0 && semanticCandidateMatchesPolarity(queryTokens, entry.Query) {
			result := lookupResultFromTimestamps(entry.ResponseBody, candidate.similarity, entry.Timestamp, entry.ExpiresAt)
			result.Query = entry.Query
			result.NegationGuard = negationGuardOutcomeFor(queryTokens, entry.Query)
			ranked.add(result)
		}
	}
	return ranked, fetchErr
}

func (h *HybridCache) recordLookupHit(
	start time.Time,
	metricOp string,
	threshold float32,
	model string,
	similarity float32,
) {
	atomic.AddInt64(&h.hitCount, 1)
	logging.Debugf("Hybrid cache hit: similarity=%.4f (threshold=%.3f)", similarity, threshold)
	logging.LogEvent("hybrid_cache_hit", map[string]interface{}{
		"backend":    "hybrid",
		"source":     "milvus",
		"similarity": similarity,
		"threshold":  threshold,
		"model":      model,
		"latency_ms": time.Since(start).Milliseconds(),
	})
	metrics.RecordCacheOperation("hybrid", metricOp, "hit_milvus", time.Since(start).Seconds())
}

func (h *HybridCache) recordLookupMiss(
	start time.Time,
	metricOp string,
	logPrefix string,
	threshold float32,
	model string,
	totalCandidates int,
	qualifiedCandidates int,
) {
	atomic.AddInt64(&h.missCount, 1)
	if qualifiedCandidates == 0 {
		if totalCandidates > 0 {
			logging.Debugf("%s: %d candidates found but none above threshold %.3f",
				logPrefix, totalCandidates, threshold)
		} else {
			logging.Debugf("%s: no candidates found in HNSW", logPrefix)
		}
	} else {
		logging.Debugf("%s: cache miss after checking %d qualified candidates", logPrefix, qualifiedCandidates)
	}
	logging.LogEvent("hybrid_cache_miss", map[string]interface{}{
		"backend":    "hybrid",
		"threshold":  threshold,
		"model":      model,
		"candidates": qualifiedCandidates,
	})
	metrics.RecordCacheOperation("hybrid", metricOp, "miss", time.Since(start).Seconds())
}
