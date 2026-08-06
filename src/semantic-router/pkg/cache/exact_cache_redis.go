package cache

import (
	"context"
	"time"
)

// FindExact returns a Redis exact-response entry without embedding inference.
func (c *RedisCache) FindExact(partition string, fingerprint string) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
	}
	responseBody, err := c.client.Get(
		context.Background(),
		exactCacheStorageKey(partition, fingerprint),
	).Bytes()
	if err != nil {
		return LookupResult{}, nil
	}
	return LookupResult{
		ResponseBody: responseBody,
		Found:        true,
		Similarity:   1,
	}, nil
}

// AddExact writes a Redis exact-response entry with the effective cache TTL.
func (c *RedisCache) AddExact(
	partition string,
	fingerprint string,
	responseBody []byte,
	ttlSeconds int,
) error {
	if !c.enabled || fingerprint == "" || ttlSeconds == 0 {
		return nil
	}
	effectiveTTL := effectiveExactTTL(ttlSeconds, c.ttlSeconds)
	expiration := time.Duration(0)
	if effectiveTTL > 0 {
		expiration = time.Duration(effectiveTTL) * time.Second
	}
	return c.client.Set(
		context.Background(),
		exactCacheStorageKey(partition, fingerprint),
		responseBody,
		expiration,
	).Err()
}
