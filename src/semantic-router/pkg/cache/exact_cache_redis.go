package cache

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// FindExact returns a Redis exact-response entry without embedding inference.
func (c *RedisCache) FindExact(ctx context.Context, partition string, fingerprint string) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	responseBody, err := c.client.Get(
		ctx,
		exactCacheStorageKey(partition, fingerprint),
	).Bytes()
	if errors.Is(err, redis.Nil) {
		return LookupResult{}, nil
	}
	if err != nil {
		return LookupResult{}, fmt.Errorf("redis exact lookup failed: %w", err)
	}
	return LookupResult{
		ResponseBody: responseBody,
		Found:        true,
		Similarity:   1,
	}, nil
}

// AddExact writes a Redis exact-response entry with the effective cache TTL.
func (c *RedisCache) AddExact(
	ctx context.Context,
	partition string,
	fingerprint string,
	responseBody []byte,
	ttlSeconds int,
) error {
	if !c.enabled || fingerprint == "" || ttlSeconds == 0 {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	effectiveTTL := effectiveExactTTL(ttlSeconds, c.ttlSeconds)
	expiration := time.Duration(0)
	if effectiveTTL > 0 {
		expiration = time.Duration(effectiveTTL) * time.Second
	}
	return c.client.Set(
		ctx,
		exactCacheStorageKey(partition, fingerprint),
		responseBody,
		expiration,
	).Err()
}
