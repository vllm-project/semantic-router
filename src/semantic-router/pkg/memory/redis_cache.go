package memory

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	defaultMemoryCacheKeyPrefix = "memory_cache:"
	defaultMemoryCacheTTL       = 300 // 5 minutes
)

// RedisCacheConfig configures the Redis hot cache for memory retrieval.
type RedisCacheConfig struct {
	Address    string
	Password   string
	DB         int
	KeyPrefix  string
	TTLSeconds int
}

// RedisCache is a Redis-backed cache for memory retrieval results.
// Key format: {keyPrefix}{userID}:{queryHash}. Cache is invalidated per user on store/update/forget.
type RedisCache struct {
	client *redis.Client
	prefix string
	ttl    time.Duration
}

// NewRedisCache creates a Redis cache client for memory retrieval.
// Returns nil if config is nil or Address is empty.
func NewRedisCache(ctx context.Context, cfg *RedisCacheConfig) (*RedisCache, error) {
	if cfg == nil || cfg.Address == "" {
		return nil, nil
	}
	prefix := cfg.KeyPrefix
	if prefix == "" {
		prefix = defaultMemoryCacheKeyPrefix
	}
	if !strings.HasSuffix(prefix, ":") {
		prefix += ":"
	}
	ttlSec := cfg.TTLSeconds
	if ttlSec <= 0 {
		ttlSec = defaultMemoryCacheTTL
	}
	opts := &redis.Options{
		Addr:     cfg.Address,
		Password: cfg.Password,
		DB:       cfg.DB,
	}
	client := redis.NewClient(opts)
	if err := client.Ping(ctx).Err(); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("redis memory cache ping failed: %w", err)
	}
	logging.Infof("Memory Redis cache: connected to %s, prefix=%s, ttl=%ds", cfg.Address, prefix, ttlSec)
	return &RedisCache{
		client: client,
		prefix: prefix,
		ttl:    time.Duration(ttlSec) * time.Second,
	}, nil
}

// cacheKey builds a key from userID and a hash of the retrieval options (query, limit, threshold, types, projectID).
func cacheKey(prefix, userID string, opts RetrieveOptions) string {
	h := sha256.New()
	h.Write([]byte(opts.Query))
	h.Write([]byte("\x00"))
	h.Write([]byte(opts.ProjectID))
	h.Write([]byte("\x00"))
	_, _ = fmt.Fprintf(h, "%d", opts.Limit)
	h.Write([]byte("\x00"))
	_, _ = fmt.Fprintf(h, "%.6f", opts.Threshold)
	for _, t := range opts.Types {
		h.Write([]byte("\x00"))
		h.Write([]byte(t))
	}
	hash := hex.EncodeToString(h.Sum(nil))[:16]
	return prefix + userID + ":" + hash
}

// Get retrieves cached results for the given options. Returns (nil, nil) on miss or error (no fatal).
func (c *RedisCache) Get(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, bool) {
	if c == nil || c.client == nil {
		return nil, false
	}
	key := cacheKey(c.prefix, opts.UserID, opts)
	val, err := c.client.Get(ctx, key).Bytes()
	if err != nil {
		if !errors.Is(err, redis.Nil) {
			logging.Debugf("Memory Redis cache get error: %v", err)
		}
		return nil, false
	}
	var results []*RetrieveResult
	if err := json.Unmarshal(val, &results); err != nil {
		logging.Warnf("Memory Redis cache: invalid cached value for %s: %v", key, err)
		return nil, false
	}
	return results, true
}

// Set stores retrieval results in the cache.
func (c *RedisCache) Set(ctx context.Context, opts RetrieveOptions, results []*RetrieveResult) {
	if c == nil || c.client == nil {
		return
	}
	key := cacheKey(c.prefix, opts.UserID, opts)
	val, err := json.Marshal(results)
	if err != nil {
		logging.Warnf("Memory Redis cache set marshal error: %v", err)
		return
	}
	if err := c.client.Set(ctx, key, val, c.ttl).Err(); err != nil {
		logging.Debugf("Memory Redis cache set error: %v", err)
	}
}

// InvalidateByUser deletes all cache entries for the given user (e.g. after store/update/forget).
func (c *RedisCache) InvalidateByUser(ctx context.Context, userID string) {
	if c == nil || c.client == nil || userID == "" {
		return
	}
	pattern := c.prefix + userID + ":*"
	iter := c.client.Scan(ctx, 0, pattern, 100).Iterator()
	var keys []string
	for iter.Next(ctx) {
		keys = append(keys, iter.Val())
	}
	if err := iter.Err(); err != nil {
		logging.Debugf("Memory Redis cache scan error: %v", err)
		return
	}
	if len(keys) == 0 {
		return
	}
	if err := c.client.Del(ctx, keys...).Err(); err != nil {
		logging.Debugf("Memory Redis cache invalidate error: %v", err)
	}
}

// Close closes the Redis client.
func (c *RedisCache) Close() error {
	if c == nil || c.client == nil {
		return nil
	}
	return c.client.Close()
}
