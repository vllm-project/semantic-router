package memory

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	defaultMemoryCacheKeyPrefix = "memory_cache:"
	defaultMemoryCacheTTL       = 300 // 5 minutes
	memoryCacheKeyVersion       = "v2:"
	// cacheWriteHold outlasts Milvus Bounded-consistency staleness (5 s), so a
	// read that still sees a just-deleted row is never written back to the cache.
	cacheWriteHold = 10 * time.Second
)

// setIfCurrentScript caches a result only if no invalidation ran since the
// caller read the user's generation and no post-invalidation hold is active.
var setIfCurrentScript = redis.NewScript(`
local generation = redis.call('GET', KEYS[1]) or '0'
if generation ~= ARGV[1] or redis.call('EXISTS', KEYS[2]) == 1 then
	return 0
end
redis.call('SET', KEYS[3], ARGV[2], 'PX', ARGV[3])
redis.call('SADD', KEYS[4], KEYS[3])
redis.call('PEXPIRE', KEYS[4], ARGV[3])
return 1
`)

// RedisCacheConfig configures the Redis hot cache for memory retrieval.
type RedisCacheConfig struct {
	Address    string
	Password   string
	DB         int
	KeyPrefix  string
	TTLSeconds int
}

// RedisCache is a Redis-backed cache for memory retrieval results.
// Value keys: {keyPrefix}v2:{userID}:{queryHash}. Each value key is also recorded
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
		Addr:                  cfg.Address,
		Password:              cfg.Password,
		DB:                    cfg.DB,
		ContextTimeoutEnabled: true,
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

// cacheKey hashes a versioned, length-delimited encoding of retrieval options.
// v2 cannot reuse legacy values that omitted retrieval policy. The user index
// deliberately stays unchanged so invalidation also removes tracked old keys.
// Only the cache identity is normalized; the backing store receives the
// caller's original options and retains its existing retrieval behavior.
func cacheKey(prefix, userID string, opts RetrieveOptions) string {
	mode := opts.HybridMode
	if !opts.HybridSearch {
		mode = "" // Both stores ignore the fusion mode for vector-only retrieval.
	} else if mode == "" {
		mode = "weighted" // vectorstore.HybridSearchConfig.applyDefaults
	}
	encoded := []byte(memoryCacheKeyVersion)
	appendField := func(value string) {
		encoded = binary.AppendUvarint(encoded, uint64(len(value)))
		encoded = append(encoded, value...)
	}
	for _, field := range []string{
		opts.Query, opts.ProjectID, strconv.Itoa(opts.Limit),
		strconv.FormatFloat(float64(opts.Threshold), 'g', -1, 32),
		strconv.FormatBool(opts.HybridSearch), mode, strconv.FormatBool(opts.AdaptiveThreshold),
		strconv.Itoa(len(opts.Types)),
	} {
		appendField(field)
	}
	for _, t := range opts.Types {
		appendField(string(t))
	}
	digest := sha256.Sum256(encoded)
	hash := hex.EncodeToString(digest[:])[:16]
	return prefix + memoryCacheKeyVersion + userID + ":" + hash
}

// userIndexKey returns the Redis set key that tracks every cached value key for a
// user. The "u:" namespace is disjoint from the versioned value-key namespace, so an
// index key can never collide with a cached value, regardless of userID content.
func (c *RedisCache) userIndexKey(userID string) string {
	return c.prefix + "u:" + userID
}

// generationKey and holdKey fence retrievals that raced an invalidation; their
// "g:" and "h:" namespaces are disjoint from value and index keys.
func (c *RedisCache) generationKey(userID string) string {
	return c.prefix + "g:" + userID
}

func (c *RedisCache) holdKey(userID string) string {
	return c.prefix + "h:" + userID
}

// generation admits a retrieval for caching and returns the user's invalidation
// generation. ok is false when the cache cannot be used safely: no user, a Redis
// error, or a post-write hold, since a read begun then may be stale when it ends.
func (c *RedisCache) generation(ctx context.Context, userID string) (string, bool) {
	if c == nil || c.client == nil || userID == "" {
		return "", false
	}
	pipe := c.client.TxPipeline()
	generationCmd := pipe.Get(ctx, c.generationKey(userID))
	holdCmd := pipe.Exists(ctx, c.holdKey(userID))
	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		logging.Debugf("Memory Redis cache generation read error: %v", err)
		return "", false
	}
	if holdCmd.Val() > 0 {
		return "", false
	}
	generation, err := generationCmd.Result()
	if errors.Is(err, redis.Nil) {
		return "0", true
	}
	if err != nil {
		logging.Debugf("Memory Redis cache generation read error: %v", err)
		return "", false
	}
	return generation, true
}

// setIfCurrent caches results read after the caller observed generation.
func (c *RedisCache) setIfCurrent(ctx context.Context, opts RetrieveOptions, generation string, results []*RetrieveResult) {
	if c == nil || c.client == nil || opts.UserID == "" {
		return
	}
	val, err := json.Marshal(results)
	if err != nil {
		logging.Warnf("Memory Redis cache set marshal error: %v", err)
		return
	}
	keys := []string{
		c.generationKey(opts.UserID), c.holdKey(opts.UserID),
		cacheKey(c.prefix, opts.UserID, opts), c.userIndexKey(opts.UserID),
	}
	if err := setIfCurrentScript.Run(ctx, c.client, keys, generation, val, c.ttl.Milliseconds()).Err(); err != nil {
		logging.Debugf("Memory Redis cache set error: %v", err)
	}
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

// Set stores retrieval results unconditionally. Retrieval paths use setIfCurrent.
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
// An error means entries may still be readable, so a caller that just committed
// a write is serving stale results until TTL - hence warn, not debug.
func (c *RedisCache) InvalidateByUser(ctx context.Context, userID string) error {
	if c == nil || c.client == nil || userID == "" {
		return nil
	}
	// Fence in-flight retrievals before deleting, so none can re-cache old results.
	pipe := c.client.TxPipeline()
	pipe.Incr(ctx, c.generationKey(userID))
	pipe.PExpire(ctx, c.generationKey(userID), 2*c.ttl+cacheWriteHold)
	pipe.Set(ctx, c.holdKey(userID), "1", cacheWriteHold)
	if _, err := pipe.Exec(ctx); err != nil {
		logging.Warnf("Memory Redis cache generation bump failed for user %s, entries remain cached until TTL: %v", userID, err)
		return err
	}
	idxKey := c.userIndexKey(userID)
	keys, err := c.client.SMembers(ctx, idxKey).Result()
	if err != nil {
		// Leave the index intact: it is the only record of this user's value keys,
		// so dropping it would strand them beyond any later invalidation.
		logging.Warnf("Memory Redis cache index read failed for user %s, entries remain cached until TTL: %v", userID, err)
		return err
	}
	// Delete the tracked value keys together with the index set itself. DEL on a
	// key that already expired via TTL is a harmless no-op, so stale index members
	// (entries whose value key already expired) never cause errors.
	if err := c.client.Del(ctx, append(keys, idxKey)...).Err(); err != nil {
		logging.Warnf("Memory Redis cache invalidate failed for user %s, %d entries remain cached until TTL: %v", userID, len(keys), err)
		return err
	}
	return nil
}

// Close closes the Redis client.
func (c *RedisCache) Close() error {
	if c == nil || c.client == nil {
		return nil
	}
	return c.client.Close()
}
