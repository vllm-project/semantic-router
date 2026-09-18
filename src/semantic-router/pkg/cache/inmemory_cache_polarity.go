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
	// Skipped means the verifier was unavailable or failed; the caller fails
	// open and serves the candidate.
	Skipped bool
}

// applyPolarityGuard runs the polarity tiers on the single winning candidate,
// cheapest first. The lexical tier (#2691) is the unconditional floor and never
// reaches the NLI tier when it already rejects, so an obvious negation or
// antonym swap costs no model call. It returns handled=true with the lookup
// outcome when the candidate must not be served.
//
// The incoming query is tokenized here, once per lookup: the guard sees only
// the winner, never every above-threshold entry in the scan.
func (c *InMemoryCache) applyPolarityGuard(
	ctx context.Context,
	start time.Time,
	model, query string,
	bestEntry CacheEntry,
	bestSimilarity, threshold float32,
) (LookupResult, bool, error) {
	if polarityMismatchTokens(tokenizeForPolarity(query), bestEntry.Query) {
		c.recordPolarityReject(start, polarityGuardTierLexical, model, query, bestEntry.Query,
			bestSimilarity, threshold, nil)
		return LookupResult{Similarity: bestSimilarity}, true, nil
	}
	return c.applyPolarityNLI(ctx, start, model, query, bestEntry, bestSimilarity, threshold)
}

// applyPolarityNLI runs the NLI tier on the winning candidate when the tier is
// enabled. It returns handled=true with the lookup outcome when the candidate
// must not be served — the request was cancelled, or the queries contradict —
// and handled=false when the caller should serve it. A rejection is a
// caller-visible miss that still reports the rejected score
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
	if !verdict.Reject {
		return LookupResult{}, false, nil
	}
	c.recordPolarityReject(start, polarityGuardTierNLI, model, query, bestEntry.Query, bestSimilarity, threshold,
		map[string]interface{}{
			"contradiction":           verdict.Contradiction,
			"contradiction_threshold": c.polarityGuard.ContradictionThreshold,
		})
	return LookupResult{Similarity: bestSimilarity}, true, nil
}

// verifyPolarityNLI runs the NLI tier on the winning candidate. It fails open:
// a missing or failing verifier serves the threshold-verified hit and records
// the skip, so a model hiccup never turns into a cache outage.
func (c *InMemoryCache) verifyPolarityNLI(ctx context.Context, model, cachedQuery, incomingQuery string) polarityNLIVerdict {
	start := time.Now()
	c.mu.RLock()
	verifier := c.polarityGuard.Verifier
	c.mu.RUnlock()
	if verifier == nil {
		c.recordPolarityNLISkipped(model, "polarity verifier not configured")
		return polarityNLIVerdict{Skipped: true}
	}

	contradiction, err := verifier(ctx, cachedQuery, incomingQuery)
	if err != nil {
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
		"backend":   "memory",
		"tier":      polarityGuardTierNLI,
		"model":     model,
		"reason":    reason,
		"fail_open": true,
	})
}

// recordPolarityReject preserves caller-visible miss semantics while emitting an
// event that distinguishes a polarity rejection from a threshold miss. detail
// carries the rejecting tier's own fields.
func (c *InMemoryCache) recordPolarityReject(
	start time.Time,
	tier string,
	model, query, cachedQuery string,
	similarity, threshold float32,
	detail map[string]interface{},
) {
	atomic.AddInt64(&c.missCount, 1)
	logging.Debugf("InMemoryCache.FindSimilarWithThreshold: POLARITY REJECT (%s) - similarity=%.4f >= threshold=%.4f; treating as miss",
		tier, similarity, threshold)
	event := map[string]interface{}{
		"backend":      "memory",
		"tier":         tier,
		"similarity":   similarity,
		"threshold":    threshold,
		"model":        model,
		"query":        logging.ContentDescriptor(query),
		"cached_query": logging.ContentDescriptor(cachedQuery),
	}
	for k, v := range detail {
		event[k] = v
	}
	logging.LogEvent("cache_negation_reject", event)
	metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
}

// SetPolarityVerifier installs this cache's own verifier without changing any
// other generation. Assembly calls it before the cache is published.
func (c *InMemoryCache) SetPolarityVerifier(verifier PolarityVerifyFunc) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.polarityGuard.Verifier = verifier
}
