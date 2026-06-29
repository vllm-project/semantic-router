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
// Value keys: {keyPrefix}v:{userID}:{queryHash}. Each value key is also recorded
// in a per-user index set ({keyPrefix}u:{userID}); the cache is invalidated per
// user on store/update/forget by deleting that set's members, so invalidation
// costs O(user's cached queries) instead of an O(keyspace) SCAN.
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

// cacheKey builds a value key from userID and a hash of the retrieval options
// (query, projectID, limit, threshold, types). Value keys live under the "v:"
// namespace so they can never collide with a per-user index key (see
// userIndexKey), regardless of userID content.
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
	return prefix + "v:" + userID + ":" + hash
}

// userIndexKey returns the Redis set key that tracks every cached value key for a
// user. The "u:" namespace is disjoint from the "v:" value-key namespace, so an
// index key can never collide with a cached value, regardless of userID content.
func (c *RedisCache) userIndexKey(userID string) string {
	return c.prefix + "u:" + userID
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
	// Cache the value and, when scoped to a user, record the key in that user's
	// index set so InvalidateByUser can find it without scanning the keyspace.
	// The index set is given the same TTL as the value and refreshed on every
	// Set, so it always outlives every live key it tracks.
	pipe := c.client.Pipeline()
	pipe.Set(ctx, key, val, c.ttl)
	if opts.UserID != "" {
		idxKey := c.userIndexKey(opts.UserID)
		pipe.SAdd(ctx, idxKey, key)
		pipe.Expire(ctx, idxKey, c.ttl)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		logging.Debugf("Memory Redis cache set error: %v", err)
	}
}

// InvalidateByUser deletes all cache entries for the given user (e.g. after store/update/forget).
// It reads the user's index set rather than scanning the keyspace, so cost is
// proportional to the number of cached queries for that user, not the total
// number of keys in Redis.
func (c *RedisCache) InvalidateByUser(ctx context.Context, userID string) {
	if c == nil || c.client == nil || userID == "" {
		return
	}
	idxKey := c.userIndexKey(userID)
	keys, err := c.client.SMembers(ctx, idxKey).Result()
	if err != nil {
		logging.Debugf("Memory Redis cache index read error: %v", err)
		return
	}
	// Delete the tracked value keys together with the index set itself. DEL on a
	// key that already expired via TTL is a harmless no-op, so stale index members
	// (entries whose value key already expired) never cause errors.
	if err := c.client.Del(ctx, append(keys, idxKey)...).Err(); err != nil {
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
