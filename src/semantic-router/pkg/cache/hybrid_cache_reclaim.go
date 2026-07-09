//go:build !windows && cgo

package cache

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	// minReclaimInterval is the floor for the background TTL-reclaim cadence.
	// A reclaim pass performs a full RebuildFromMilvus, so we never run it more
	// often than this regardless of how short the configured TTL is.
	minReclaimInterval = 60 * time.Second

	// reclaimTimeout bounds a single background reclaim pass. A pass runs a full
	// RebuildFromMilvus, which holds the cache write lock for its whole body, so
	// this is also the worst-case window during which lookups/writes are blocked
	// by reclamation. It matches the startup-rebuild timeout for consistency.
	reclaimTimeout = 5 * time.Minute
)

// reclaimIntervalForTTL derives the background reclaim cadence from the TTL.
// We re-sync roughly once per TTL window (so an expired entry is dropped within
// about one extra TTL period), floored at minReclaimInterval to avoid hammering
// Milvus with full rebuilds when the TTL is very short.
func reclaimIntervalForTTL(ttlSeconds int) time.Duration {
	d := time.Duration(ttlSeconds) * time.Second
	if d < minReclaimInterval {
		return minReclaimInterval
	}
	return d
}

// startBackgroundReclaim launches the periodic TTL-reclamation goroutine.
//
// The hybrid backend keeps an in-process HNSW index alongside Milvus. Milvus
// expires entries via its own TTL and silently deletes them, but the in-memory
// index is otherwise only trimmed when it reaches maxMemoryEntries. Without this
// goroutine, entries that Milvus has already expired keep occupying in-process
// memory (and HNSW search slots) until the capacity cap forces FIFO eviction
// (see https://github.com/vllm-project/semantic-router/issues/2288).
//
// Reclamation reuses RebuildFromMilvus: re-querying Milvus returns only the live
// (non-expired) entries, so the rebuilt index naturally drops the expired ones.
// It is a no-op when TTL is disabled (ttlSeconds <= 0), since nothing expires.
func (h *HybridCache) startBackgroundReclaim() {
	if !h.enabled || h.ttlSeconds <= 0 {
		return
	}

	if h.reclaimFn == nil {
		h.reclaimFn = h.RebuildFromMilvus
	}

	interval := reclaimIntervalForTTL(h.ttlSeconds)
	h.stopReclaim = make(chan struct{})
	h.reclaimDone = make(chan struct{})
	h.reclaimTicker = time.NewTicker(interval)

	logging.ComponentEvent("cache", "hybrid_cache_reclaim_started", map[string]interface{}{
		"interval_seconds": interval.Seconds(),
		"ttl_seconds":      h.ttlSeconds,
	})

	go func() {
		defer close(h.reclaimDone)
		h.backgroundReclaim()
	}()
}

// backgroundReclaim reclaims entries that Milvus has expired on each tick until
// the cache is closed. A single pass is best-effort: failures are logged and the
// loop continues.
func (h *HybridCache) backgroundReclaim() {
	for {
		select {
		case <-h.reclaimTicker.C:
			ctx, cancel := context.WithTimeout(context.Background(), reclaimTimeout)
			if err := h.reclaimFn(ctx); err != nil {
				logging.ComponentWarnEvent("cache", "hybrid_cache_reclaim_failed", map[string]interface{}{
					"error": err.Error(),
				})
			}
			cancel()
		case <-h.stopReclaim:
			return
		}
	}
}

// stopBackgroundReclaim signals the reclaim goroutine to exit, stops its ticker,
// and blocks until the goroutine has actually returned. Blocking matters: it is
// called from Close() before the write lock is taken, so a reclaim pass already
// in flight can still acquire h.mu, finish, and exit — guaranteeing no pass is
// running by the time Close() nils out the in-memory structures (no
// use-after-close). Safe to call multiple times and when reclaim was never
// started.
func (h *HybridCache) stopBackgroundReclaim() {
	h.closeOnce.Do(func() {
		if h.stopReclaim != nil {
			close(h.stopReclaim)
		}
		if h.reclaimTicker != nil {
			h.reclaimTicker.Stop()
		}
		if h.reclaimDone != nil {
			<-h.reclaimDone
			logging.ComponentEvent("cache", "hybrid_cache_reclaim_stopped", map[string]interface{}{
				"ttl_seconds": h.ttlSeconds,
			})
		}
	})
}
