//go:build !windows && cgo

package cache

import (
	"context"
	"time"
)

type exactMemoryEntry struct {
	responseBody []byte
	storedAt     time.Time
	expiresAt    time.Time
}

// FindExact returns a response without running embedding inference.
func (c *InMemoryCache) FindExact(ctx context.Context, partition string, fingerprint string) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
	}
	if err := ctxErr(ctx); err != nil {
		return LookupResult{}, err
	}
	key := exactCacheStorageKey(partition, fingerprint)
	c.mu.RLock()
	entry, ok := c.exactEntries[key]
	c.mu.RUnlock()
	if !ok {
		return LookupResult{}, nil
	}
	if !entry.expiresAt.IsZero() && time.Now().After(entry.expiresAt) {
		c.mu.Lock()
		delete(c.exactEntries, key)
		c.mu.Unlock()
		return LookupResult{}, nil
	}
	return lookupResultFromTimestamps(append([]byte(nil), entry.responseBody...), 1, entry.storedAt, entry.expiresAt), nil
}

// AddExact stores a complete exact response under the normalized request hash.
// It rechecks cancellation under the lock so cancelled writes publish nothing.
func (c *InMemoryCache) AddExact(
	ctx context.Context,
	partition string,
	fingerprint string,
	responseBody []byte,
	ttlSeconds int,
) error {
	if !c.enabled || fingerprint == "" || ttlSeconds == 0 {
		return nil
	}
	if err := ctxErr(ctx); err != nil {
		return err
	}
	effectiveTTL := effectiveExactTTL(ttlSeconds, c.ttlSeconds)
	entry := exactMemoryEntry{
		responseBody: append([]byte(nil), responseBody...),
		storedAt:     time.Now(),
	}
	if effectiveTTL > 0 {
		entry.expiresAt = time.Now().Add(time.Duration(effectiveTTL) * time.Second)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := ctxErr(ctx); err != nil {
		return err
	}
	if c.maxEntries > 0 && len(c.exactEntries) >= c.maxEntries {
		for key := range c.exactEntries {
			delete(c.exactEntries, key)
			break
		}
	}
	c.exactEntries[exactCacheStorageKey(partition, fingerprint)] = entry
	return nil
}
