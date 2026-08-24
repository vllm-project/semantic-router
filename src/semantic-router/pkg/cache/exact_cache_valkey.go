package cache

import (
	"context"
	"fmt"
)

// FindExact returns a Valkey exact-response entry without embedding inference.
func (c *ValkeyCache) FindExact(ctx context.Context, partition string, fingerprint string) (LookupResult, error) {
	if !c.enabled || fingerprint == "" {
		return LookupResult{}, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	raw, err := c.client.CustomCommand(
		ctx,
		[]string{"GET", exactCacheStorageKey(partition, fingerprint)},
	)
	if err != nil {
		return LookupResult{}, fmt.Errorf("valkey exact lookup failed: %w", err)
	}
	if raw == nil {
		return LookupResult{}, nil
	}
	var responseBody []byte
	switch value := raw.(type) {
	case string:
		responseBody = []byte(value)
	case []byte:
		responseBody = append([]byte(nil), value...)
	default:
		responseBody = []byte(fmt.Sprint(value))
	}
	if len(responseBody) == 0 {
		return LookupResult{}, nil
	}
	return LookupResult{
		ResponseBody: responseBody,
		Found:        true,
		Similarity:   1,
	}, nil
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
	command := []string{
		"SET",
		exactCacheStorageKey(partition, fingerprint),
		string(responseBody),
	}
	if effectiveTTL > 0 {
		command = append(command, "EX", fmt.Sprintf("%d", effectiveTTL))
	}
	_, err := c.client.CustomCommand(ctx, command)
	return err
}
