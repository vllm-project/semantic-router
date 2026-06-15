//go:build !windows && cgo

package cache

import (
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Close releases all resources held by the cache
func (c *InMemoryCache) Close() error {
	// Use sync.Once to ensure cleanup happens only once
	c.closeOnce.Do(func() {
		// Stop background cleanup goroutine
		if c.stopCleanup != nil {
			close(c.stopCleanup)
		}
		if c.cleanupTicker != nil {
			c.cleanupTicker.Stop()
		}

		c.mu.Lock()
		defer c.mu.Unlock()

		// Clear all entries to free memory
		c.entries = nil

		// Zero cache entries metrics
		metrics.UpdateCacheEntries("memory", 0)
	})

	return nil
}

// backgroundCleanup runs periodic cleanup of expired entries
func (c *InMemoryCache) backgroundCleanup() {
	for {
		select {
		case <-c.cleanupTicker.C:
			c.mu.Lock()
			c.cleanupExpiredEntries()
			c.mu.Unlock()
		case <-c.stopCleanup:
			return
		}
	}
}

// GetStats provides current cache performance metrics
func (c *InMemoryCache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	hits := atomic.LoadInt64(&c.hitCount)
	misses := atomic.LoadInt64(&c.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	stats := CacheStats{
		TotalEntries: len(c.entries),
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}

	if c.lastCleanupTime != nil {
		stats.LastCleanupTime = c.lastCleanupTime
	}

	return stats
}

// cleanupExpiredEntries removes entries that have exceeded their TTL and immediately rebuilds HNSW if HNSW is enabled and cleanup occurs.
//
// Caller must hold a write lock.
func (c *InMemoryCache) cleanupExpiredEntries() {
	c.cleanupExpiredEntriesInternal(false)
}

// cleanupExpiredEntriesDeferred removes expired entries.
//
// If HNSW is enabled and cleanup occurs, it marks HNSW as needing rebuild but defers the rebuild until next call to addEntryToHNSWIndex or rebuildHNSWIndex.
// This is used in write paths that already plan to mutate the slice again (evictions, appends) so we only rebuild once per batch.
//
// Caller must hold a write lock.
func (c *InMemoryCache) cleanupExpiredEntriesDeferred() {
	c.cleanupExpiredEntriesInternal(true)
}

// cleanupExpiredEntriesInternal optionally postpones HNSW rebuild until the caller finishes batching updates.
// Uses expiration heap for O(k) cleanup where k = number of expired entries.
func (c *InMemoryCache) cleanupExpiredEntriesInternal(deferRebuild bool) {
	if c.ttlSeconds <= 0 {
		return
	}

	now := time.Now()

	// Use expiration heap for efficient O(k) cleanup where k = expired entries
	expiredRequestIDs := c.expirationHeap.PopExpired(now)

	if len(expiredRequestIDs) == 0 {
		return
	}

	// Remove expired entries from the entries slice
	// Build a set of expired request IDs for O(1) lookup
	expiredSet := make(map[string]bool, len(expiredRequestIDs))
	for _, id := range expiredRequestIDs {
		expiredSet[id] = true
	}

	// Compact the entries slice, keeping non-expired entries
	writeIdx := 0
	for readIdx := 0; readIdx < len(c.entries); readIdx++ {
		entry := c.entries[readIdx]
		if !expiredSet[entry.RequestID] {
			if writeIdx != readIdx {
				c.entries[writeIdx] = entry
				// Update tracking for the moved entry
				c.updateMovedEntryIndex(entry.RequestID, readIdx, writeIdx)
			}
			writeIdx++
		} else {
			// Remove from tracking structures
			c.removeEntryFromTracking(readIdx, entry.RequestID)
		}
	}
	c.entries = c.entries[:writeIdx]

	expiredCount := len(expiredRequestIDs)
	logging.Debugf("InMemoryCache: TTL cleanup removed %d expired entries (remaining: %d)",
		expiredCount, len(c.entries))
	logging.LogEvent("cache_cleanup", map[string]interface{}{
		"backend":         "memory",
		"expired_count":   expiredCount,
		"remaining_count": len(c.entries),
		"ttl_seconds":     c.ttlSeconds,
	})
	cleanupTime := time.Now()
	c.lastCleanupTime = &cleanupTime

	// Record cleanup operation metric
	if expiredCount > 0 {
		metrics.RecordCacheOperation("memory", "cleanup_expired", "success", time.Since(now).Seconds())
	}

	// Rebuild HNSW index if entries were removed and deferRebuild is false
	if expiredCount > 0 && c.useHNSW && c.hnswIndex != nil {
		logging.Debugf("InMemoryCache: TTL cleanup removed entries, marking HNSW index as needing rebuild")
		c.hnswNeedsRebuild = true
		c.hnswIndex.markStale()
		if !deferRebuild {
			c.rebuildHNSWIndex()
		}
	}

	// Update metrics after cleanup
	metrics.UpdateCacheEntries("memory", len(c.entries))
}

// isExpired checks if a cache entry has expired based on its last access time
func (c *InMemoryCache) isExpired(entry CacheEntry, now time.Time) bool {
	// Check per-entry expiration first
	if !entry.ExpiresAt.IsZero() {
		return now.After(entry.ExpiresAt)
	}

	// Fall back to global TTL for backward compatibility
	if c.ttlSeconds <= 0 {
		return false
	}

	return now.Sub(entry.LastAccessAt) >= time.Duration(c.ttlSeconds)*time.Second
}

// updateAccessInfo updates the access information for the given entry index
func (c *InMemoryCache) updateAccessInfo(entryIndex int, target CacheEntry) {
	now := time.Now()

	// fast path
	if entryIndex < len(c.entries) && c.entries[entryIndex].RequestID == target.RequestID {
		c.entries[entryIndex].LastAccessAt = now
		c.entries[entryIndex].HitCount++

		// Update optimized eviction policy tracking
		c.notifyAccessToEvictionPolicy(entryIndex, target.RequestID)

		// Extend TTL in expiration heap (sliding window TTL)
		// Use per-entry TTL if set, otherwise use global TTL
		effectiveTTL := c.ttlSeconds
		if c.entries[entryIndex].TTLSeconds > 0 {
			effectiveTTL = c.entries[entryIndex].TTLSeconds
		}
		if effectiveTTL > 0 {
			newExpiresAt := now.Add(time.Duration(effectiveTTL) * time.Second)
			c.entries[entryIndex].ExpiresAt = newExpiresAt
			c.expirationHeap.UpdateExpiration(target.RequestID, newExpiresAt)
		}
		return
	}

	// fallback to linear search
	for i := range c.entries {
		if c.entries[i].RequestID == target.RequestID {
			c.entries[i].LastAccessAt = now
			c.entries[i].HitCount++

			// Update optimized eviction policy tracking
			c.notifyAccessToEvictionPolicy(i, target.RequestID)

			// Extend TTL in expiration heap (sliding window TTL)
			// Use per-entry TTL if set, otherwise use global TTL
			effectiveTTL := c.ttlSeconds
			if c.entries[i].TTLSeconds > 0 {
				effectiveTTL = c.entries[i].TTLSeconds
			}
			if effectiveTTL > 0 {
				newExpiresAt := now.Add(time.Duration(effectiveTTL) * time.Second)
				c.entries[i].ExpiresAt = newExpiresAt
				c.expirationHeap.UpdateExpiration(target.RequestID, newExpiresAt)
			}
			break
		}
	}
}

// addEntryToHNSWIndex adds a new entry to the HNSW index, rebuilding if hnswNeedsRebuild is true.
// If HNSW is disabled, this is a no-op.
//
// Caller must hold a write lock.
func (c *InMemoryCache) addEntryToHNSWIndex(entryIndex int, embedding []float32) {
	if !c.useHNSW || c.hnswIndex == nil {
		return
	}

	if c.hnswNeedsRebuild {
		logging.Debugf("InMemoryCache.addEntryToHNSWIndex: HNSW index marked as needing rebuild, rebuilding now")
		c.rebuildHNSWIndex() // Rebuild HNSW index if stale
	} else {
		logging.Debugf("InMemoryCache.addEntryToHNSWIndex: adding new node to HNSW index for entryIndex=%d", entryIndex)
		c.hnswIndex.addNode(entryIndex, embedding, c.entries) // Not stale, just add the new node
	}
}

// evictOne removes one entry based on the configured eviction policy.
// It marks HNSW as needing rebuild if HNSW is enabled and an eviction occurs. HNSW will be rebuilt on next call to addEntryToHNSWIndex or rebuildHNSWIndex.
//
// Caller must hold a write lock.
func (c *InMemoryCache) evictOne() {
	if len(c.entries) == 0 {
		return
	}

	// Use optimized O(1) eviction
	victimIdx := c.evictUsingOptimizedPolicy()
	if victimIdx < 0 || victimIdx >= len(c.entries) {
		// Fallback to legacy O(n) eviction if optimized policy is not available
		victimIdx = c.evictionPolicy.SelectVictim(c.entries)
		if victimIdx < 0 || victimIdx >= len(c.entries) {
			return
		}
	}

	evictedRequestID := c.entries[victimIdx].RequestID

	// If using HNSW, we need to rebuild the index after eviction
	// For simplicity, we'll mark that a rebuild is needed
	if c.useHNSW && c.hnswIndex != nil {
		logging.Debugf("InMemoryCache.evictOne: HNSW index marked as needing rebuild due to eviction")
		// Note: HNSW doesn't support efficient deletion, leave the rebuild for the next insertion so we only rebuild once for eviction + append.
		c.hnswNeedsRebuild = true
		c.hnswIndex.markStale()
	}

	// Remove from optimized tracking structures
	c.removeEntryFromTracking(victimIdx, evictedRequestID)

	// Swap with last entry and shrink slice
	lastIdx := len(c.entries) - 1
	if victimIdx != lastIdx {
		movedEntry := c.entries[lastIdx]
		c.entries[victimIdx] = movedEntry
		// Update tracking for the moved entry
		c.updateMovedEntryIndex(movedEntry.RequestID, lastIdx, victimIdx)
	}
	c.entries = c.entries[:lastIdx]

	logging.LogEvent("cache_evicted", map[string]any{
		"backend":     "memory",
		"request_id":  evictedRequestID,
		"max_entries": c.maxEntries,
	})

	// Record eviction metric
	metrics.RecordCacheOperation("memory", "evict", "success", 0)

	// Update cache entries count after eviction
	metrics.UpdateCacheEntries("memory", len(c.entries))
}

// ===== Optimized Eviction Policy Helpers =====

// registerEntryWithEvictionPolicy registers a new entry with the appropriate optimized eviction policy.
// Caller must hold a write lock.
func (c *InMemoryCache) registerEntryWithEvictionPolicy(entryIndex int, requestID string) {
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			c.optimizedLRU.OnInsert(entryIndex, requestID)
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			c.optimizedLFU.OnInsert(entryIndex, requestID)
		}
	default: // FIFO
		if c.optimizedFIFO != nil {
			c.optimizedFIFO.OnInsert(entryIndex, requestID)
		}
	}
}

// evictUsingOptimizedPolicy uses the optimized O(1) eviction policy to select and evict a victim.
// Returns the victim index, or -1 if no victim was evicted.
// Caller must hold a write lock.
func (c *InMemoryCache) evictUsingOptimizedPolicy() int {
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			return c.optimizedLRU.Evict()
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			return c.optimizedLFU.Evict()
		}
	default: // FIFO
		if c.optimizedFIFO != nil {
			return c.optimizedFIFO.Evict()
		}
	}
	return -1
}

// removeEntryFromTracking removes an entry from all tracking structures.
// Caller must hold a write lock.
func (c *InMemoryCache) removeEntryFromTracking(entryIndex int, requestID string) {
	// Remove from entryMap
	delete(c.entryMap, requestID)

	// Remove from expiration heap
	c.expirationHeap.Remove(requestID)

	// Remove from optimized eviction policy
	// Note: This is idempotent - safe to call even if already removed by Evict()
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			c.optimizedLRU.OnRemove(entryIndex, requestID)
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			c.optimizedLFU.OnRemove(entryIndex, requestID)
		}
	default: // FIFO
		if c.optimizedFIFO != nil {
			c.optimizedFIFO.OnRemove(entryIndex, requestID)
		}
	}
}

// updateMovedEntryIndex updates tracking structures when an entry is moved to a new index.
// This is called after swap operations during eviction or cleanup.
// Caller must hold a write lock.
func (c *InMemoryCache) updateMovedEntryIndex(requestID string, oldIdx, newIdx int) {
	// Update entryMap
	c.entryMap[requestID] = newIdx

	// Update expiration heap
	c.expirationHeap.UpdateIndex(requestID, newIdx)

	// Update optimized eviction policy
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			c.optimizedLRU.UpdateIndex(requestID, oldIdx, newIdx)
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			c.optimizedLFU.UpdateIndex(requestID, oldIdx, newIdx)
		}
	default: // FIFO
		if c.optimizedFIFO != nil {
			c.optimizedFIFO.UpdateIndex(requestID, oldIdx, newIdx)
		}
	}
}

// notifyAccessToEvictionPolicy notifies the eviction policy of an access to update recency/frequency.
// Caller must hold a write lock.
func (c *InMemoryCache) notifyAccessToEvictionPolicy(entryIndex int, requestID string) {
	switch c.evictionPolicyType {
	case LRUEvictionPolicyType:
		if c.optimizedLRU != nil {
			c.optimizedLRU.OnAccess(entryIndex, requestID)
		}
	case LFUEvictionPolicyType:
		if c.optimizedLFU != nil {
			c.optimizedLFU.OnAccess(entryIndex, requestID)
		}
		// FIFO doesn't need access tracking
	}
}
