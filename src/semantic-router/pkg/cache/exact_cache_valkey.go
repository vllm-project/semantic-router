//go:build !windows && cgo && !riscv64

package cache

import (
	"context"
	"fmt"
	"time"
)

func (c *ValkeyCache) fallbackGetExact(ctx context.Context, key string) (LookupResult, bool) {
	rawStr, getErr := c.client.CustomCommand(ctx, []string{"GET", key})
	if getErr == nil && rawStr != nil {
		responseBody := valkeyFallbackBytes(rawStr)
		if len(responseBody) > 0 {
			return LookupResult{
				ResponseBody: responseBody,
				Found:        true,
				Similarity:   1,
			}, true
		}
	}
	return LookupResult{}, false
}

// FindExact returns a Valkey exact-response entry without embedding inference.
func (c *ValkeyCache) FindExact(ctx context.Context, partition string, fingerprint string) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	key := exactCacheStorageKey(partition, fingerprint)
	raw, err := c.client.CustomCommand(ctx, []string{"HGETALL", key})
	if err != nil {
		if res, ok := c.fallbackGetExact(ctx, key); ok {
			return res, nil
		}
		return LookupResult{}, fmt.Errorf("valkey exact lookup failed: %w", err)
	}
	if raw == nil {
		return LookupResult{}, nil
	}
	fields := parseValkeyHashFields(raw)
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

// AddExact writes a Valkey exact-response entry with the effective cache TTL.
func (c *ValkeyCache) AddExact(
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
	var expiresAt int64
	if effectiveTTL > 0 {
		expiresAt = now.Add(time.Duration(effectiveTTL) * time.Second).Unix()
	}
	delCmd := []string{"DEL", key}
	_, _ = c.client.CustomCommand(ctx, delCmd)
	command := []string{
		"HSET",
		key,
		"response_body", string(responseBody),
		"timestamp", fmt.Sprintf("%d", now.Unix()),
		"expires_at", fmt.Sprintf("%d", expiresAt),
		"ttl_seconds", fmt.Sprintf("%d", effectiveTTL),
	}
	if _, err := c.client.CustomCommand(ctx, command); err != nil {
		return err
	}
	if effectiveTTL > 0 {
		expireCmd := []string{"EXPIRE", key, fmt.Sprintf("%d", effectiveTTL)}
		_, err := c.client.CustomCommand(ctx, expireCmd)
		return err
	}
	return nil
}
