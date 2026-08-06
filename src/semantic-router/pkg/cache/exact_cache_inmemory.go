//go:build !windows && cgo

package cache

import "time"

type exactMemoryEntry struct {
	responseBody []byte
	expiresAt    time.Time
}

// FindExact returns a response without running embedding inference.
func (c *InMemoryCache) FindExact(partition string, fingerprint string) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
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
	return LookupResult{
		ResponseBody: append([]byte(nil), entry.responseBody...),
		Found:        true,
		Similarity:   1,
	}, nil
}

// AddExact stores a complete exact response under the normalized request hash.
func (c *InMemoryCache) AddExact(
	partition string,
	fingerprint string,
	responseBody []byte,
	ttlSeconds int,
) error {
	if !c.enabled || fingerprint == "" || ttlSeconds == 0 {
		return nil
	}
	effectiveTTL := effectiveExactTTL(ttlSeconds, c.ttlSeconds)
	entry := exactMemoryEntry{responseBody: append([]byte(nil), responseBody...)}
	if effectiveTTL > 0 {
		entry.expiresAt = time.Now().Add(time.Duration(effectiveTTL) * time.Second)
	}
	c.mu.Lock()
	if c.maxEntries > 0 && len(c.exactEntries) >= c.maxEntries {
		for key := range c.exactEntries {
			delete(c.exactEntries, key)
			break
		}
	}
	c.exactEntries[exactCacheStorageKey(partition, fingerprint)] = entry
	c.mu.Unlock()
	return nil
}
