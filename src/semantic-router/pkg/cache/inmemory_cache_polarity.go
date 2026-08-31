//go:build !windows && cgo

package cache

import (
	"context"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// polarityNLIVerdict is the outcome of one NLI verification of a cache candidate.
type polarityNLIVerdict struct {
	Reject        bool
	Contradiction float32
	// Skipped means the verifier was unavailable or failed; the caller treats
	// the unverified semantic candidate as a cache miss.
	Skipped bool
}

// applyPolarityNLI runs the NLI tier on the winning candidate when the tier is
// enabled. It returns handled=true with the lookup outcome when the candidate
// must not be served: the request was cancelled, the verifier could not verify
// the candidate, or the queries contradict. It returns handled=false when the
// caller should serve the candidate. An unverified or rejected candidate is a
// caller-visible miss that still reports the candidate score
// (LookupResult.Similarity) so near-threshold rejections stay diagnosable.
func (c *InMemoryCache) applyPolarityNLI(
	ctx context.Context,
	start time.Time,
	model, query string,
	bestEntry CacheEntry,
	bestSimilarity, threshold float32,
) (LookupResult, bool, error) {
	if !c.polarityGuard.UseNLI {
		return LookupResult{}, false, nil
	}
	if err := ctxErr(ctx); err != nil {
		metrics.RecordCacheOperation("memory", "find_similar", "canceled", time.Since(start).Seconds())
		return LookupResult{}, true, err
	}
	verdict := c.verifyPolarityNLI(ctx, model, bestEntry.Query, query)
	if err := ctxErr(ctx); err != nil {
		metrics.RecordCacheOperation("memory", "find_similar", "canceled", time.Since(start).Seconds())
		return LookupResult{}, true, err
	}
	if verdict.Skipped {
		c.recordPolaritySkippedMiss(start, bestSimilarity)
		return LookupResult{Similarity: bestSimilarity}, true, nil
	}
	if !verdict.Reject {
		return LookupResult{}, false, nil
	}
	c.recordPolarityReject(start, model, query, bestEntry.Query, bestSimilarity, threshold, verdict)
	return LookupResult{Similarity: bestSimilarity}, true, nil
}

// verifyPolarityNLI runs the NLI tier on the winning candidate. A missing or
// failing verifier records a skip; the caller degrades that unverified semantic
// hit to a cache miss so the request can continue to the model backend.
func (c *InMemoryCache) verifyPolarityNLI(ctx context.Context, model, cachedQuery, incomingQuery string) polarityNLIVerdict {
	start := time.Now()
	verifier := loadPolarityVerifier()
	if verifier == nil {
		c.recordPolarityNLISkipped(model, "polarity verifier not configured")
		return polarityNLIVerdict{Skipped: true}
	}

	contradiction, err := verifier(ctx, cachedQuery, incomingQuery)
	if err != nil {
		// Cancellation belongs to the lookup lifecycle, not verifier health. The
		// caller observes ctx.Err() after this function returns.
		if ctxErr(ctx) != nil {
			return polarityNLIVerdict{}
		}
		metrics.RecordCacheOperation("memory", "polarity_nli", "error", time.Since(start).Seconds())
		c.recordPolarityNLISkipped(model, err.Error())
		return polarityNLIVerdict{Skipped: true}
	}

	if contradiction > c.polarityGuard.ContradictionThreshold {
		metrics.RecordCacheOperation("memory", "polarity_nli", "reject", time.Since(start).Seconds())
		return polarityNLIVerdict{Reject: true, Contradiction: contradiction}
	}
	metrics.RecordCacheOperation("memory", "polarity_nli", "pass", time.Since(start).Seconds())
	return polarityNLIVerdict{Contradiction: contradiction}
}

func (c *InMemoryCache) recordPolarityNLISkipped(model, reason string) {
	logging.ComponentWarnEvent("cache", "cache_polarity_nli_skipped", map[string]interface{}{
		"backend":    "memory",
		"tier":       polarityGuardTierNLI,
		"model":      model,
		"reason":     reason,
		"cache_miss": true,
	})
}

// recordPolaritySkippedMiss keeps verifier failures out of the contradiction
// metrics while preserving the normal cache-miss accounting used by callers.
func (c *InMemoryCache) recordPolaritySkippedMiss(start time.Time, similarity float32) {
	atomic.AddInt64(&c.missCount, 1)
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: POLARITY VERIFICATION SKIPPED - similarity=%.4f; treating as miss", similarity)
	metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
}

// recordPolarityReject preserves caller-visible miss semantics while emitting an
// event that distinguishes a polarity rejection from a threshold miss.
func (c *InMemoryCache) recordPolarityReject(
	start time.Time,
	model, query, cachedQuery string,
	similarity, threshold float32,
	verdict polarityNLIVerdict,
) {
	atomic.AddInt64(&c.missCount, 1)
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: POLARITY REJECT (nli) - similarity=%.4f >= threshold=%.4f but contradiction=%.4f > %.4f; treating as miss",
		similarity, threshold, verdict.Contradiction, c.polarityGuard.ContradictionThreshold)
	logging.LogEvent("cache_negation_reject", map[string]interface{}{
		"backend":                 "memory",
		"tier":                    polarityGuardTierNLI,
		"similarity":              similarity,
		"threshold":               threshold,
		"contradiction":           verdict.Contradiction,
		"contradiction_threshold": c.polarityGuard.ContradictionThreshold,
		"model":                   model,
		"query":                   logging.ContentDescriptor(query),
		"cached_query":            logging.ContentDescriptor(cachedQuery),
	})
	metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
}
