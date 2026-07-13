//go:build !windows && cgo

package cache

import (
	"context"
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
func (h *HybridCache) FindSimilar(ctx context.Context, model string, query string) (LookupResult, error) {
	return h.findSimilar(ctx, model, query, h.similarityThreshold, "find_similar", "HybridCache.FindSimilar")
}

// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold.
func (h *HybridCache) FindSimilarWithThreshold(ctx context.Context, model string, query string, threshold float32) (LookupResult, error) {
	return h.findSimilar(ctx, model, query, threshold, "find_similar_threshold", "HybridCache.FindSimilarWithThreshold")
}

func (h *HybridCache) findSimilar(
	ctx context.Context,
	model string,
	query string,
	threshold float32,
	metricOp string,
	logPrefix string,
) (LookupResult, error) {
	start := time.Now()

	if !h.enabled {
		return LookupResult{}, nil
	}

	logging.Debugf("%s: searching for model='%s', query='%s', threshold=%.3f",
		logPrefix, model, shortenHybridQuery(query), threshold)

	queryEmbedding, err := h.generateEmbedding(ctx, query)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", metricOp, "error", time.Since(start).Seconds())
		return LookupResult{}, fmt.Errorf("failed to generate embedding: %w", err)
	}

	candidatesWithIDs, totalCandidates := h.searchCandidates(queryEmbedding, threshold)
	if len(candidatesWithIDs) == 0 {
		h.recordLookupMiss(start, metricOp, logPrefix, threshold, model, totalCandidates, 0)
		return LookupResult{}, nil
	}

	logging.Debugf("%s: HNSW returned %d candidates, %d above threshold",
		logPrefix, totalCandidates, len(candidatesWithIDs))

	responseBody, candidate, found := h.fetchResponseFromCandidates(ctx, logPrefix, candidatesWithIDs)
	if found {
		h.recordLookupHit(start, metricOp, threshold, model, candidate)
		return LookupResult{Body: responseBody, Found: true, Similarity: candidate.similarity}, nil
	}

	h.recordLookupMiss(start, metricOp, logPrefix, threshold, model, totalCandidates, len(candidatesWithIDs))
	// Best HNSW candidate was above threshold but Milvus fetch failed;
	// surface its score as the per-request best so callers can distinguish this
	// "found in index, unfetchable" miss from a no-candidate miss.
	return LookupResult{Similarity: candidatesWithIDs[0].similarity}, nil
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

func (h *HybridCache) fetchResponseFromCandidates(
	parent context.Context,
	logPrefix string,
	candidates []candidateWithID,
) ([]byte, candidateWithID, bool) {
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()

	for _, candidate := range candidates {
		fetchCtx, fetchCancel := context.WithTimeout(ctx, 2*time.Second)
		responseBody, err := h.milvusCache.GetByID(fetchCtx, candidate.milvusID)
		fetchCancel()
		if err != nil {
			logging.Debugf("%s: Milvus GetByID failed for %s: %v", logPrefix, candidate.milvusID, err)
			continue
		}
		if responseBody != nil {
			return responseBody, candidate, true
		}
	}

	return nil, candidateWithID{}, false
}

func (h *HybridCache) recordLookupHit(
	start time.Time,
	metricOp string,
	threshold float32,
	model string,
	candidate candidateWithID,
) {
	atomic.AddInt64(&h.hitCount, 1)
	logging.Debugf("Hybrid cache hit: similarity=%.4f (threshold=%.3f)", candidate.similarity, threshold)
	logging.LogEvent("hybrid_cache_hit", map[string]interface{}{
		"backend":    "hybrid",
		"source":     "milvus",
		"similarity": candidate.similarity,
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

func shortenHybridQuery(query string) string {
	if len(query) <= 50 {
		return query
	}
	return query[:50] + "..."
}
