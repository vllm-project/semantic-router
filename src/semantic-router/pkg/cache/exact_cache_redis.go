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
	key := exactCacheStorageKey(partition, fingerprint)
	fields, err := c.client.HGetAll(ctx, key).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		// Fallback for legacy plain-string keys if present
		if responseBody, getErr := c.client.Get(ctx, key).Bytes(); getErr == nil && len(responseBody) > 0 {
			return LookupResult{
				ResponseBody: responseBody,
				Found:        true,
				Similarity:   1,
			}, nil
		}
		return LookupResult{}, fmt.Errorf("redis exact lookup failed: %w", err)
	}
	if len(fields) == 0 {
		return LookupResult{}, nil
	}
	responseBodyStr, exists := fields["response_body"]
	if !exists || responseBodyStr == "" {
		return LookupResult{}, nil
	}
	storedAt, expiresAt := parseExactTimingFields(fields)
	return lookupResultFromTimestamps([]byte(responseBodyStr), 1.0, storedAt, expiresAt), nil
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
	key := exactCacheStorageKey(partition, fingerprint)
	now := time.Now()
	var expiresAt time.Time
	if effectiveTTL > 0 {
		expiresAt = now.Add(time.Duration(effectiveTTL) * time.Second)
	}
	hashFields := map[string]interface{}{
		"response_body": string(responseBody),
		"timestamp":     now.Unix(),
		"expires_at":    expiresAt.Unix(),
		"ttl_seconds":   effectiveTTL,
	}
	if err := c.client.HSet(ctx, key, hashFields).Err(); err != nil {
		return err
	}
	if effectiveTTL > 0 {
		return c.client.Expire(ctx, key, time.Duration(effectiveTTL)*time.Second).Err()
	}
	return nil
}
