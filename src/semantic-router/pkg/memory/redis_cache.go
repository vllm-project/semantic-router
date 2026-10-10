package memory

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
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
	memoryCacheKeyVersion       = "v3:"
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
// Value keys use a versioned namespace: {keyPrefix}v3:{userID}:{queryHash}.
// Each value key is also recorded in a per-user index set
// ({keyPrefix}u:{userID}); the cache is invalidated per user on
// store/update/forget by deleting that set's members, so invalidation costs
// O(user's cached queries) instead of an O(keyspace) SCAN. The index namespace
// is shared across value-key versions so it can clean up entries left by older
// instances during a rolling upgrade.
type RedisCache struct {
	client *redis.Client
	prefix string
	ttl    time.Duration
}

// readCacheEntryScript reads the value and its refill generation atomically.
// A short-lived random generation token is created when needed. Its expiry is
// safe: conditional writes reject a missing token, so an old in-flight query
// cannot mistake an expired generation for a fresh one.
var readCacheEntryScript = redis.NewScript(`
local generation = redis.call("GET", KEYS[2])
if not generation then
  generation = ARGV[1]
  redis.call("SET", KEYS[2], generation, "PX", ARGV[2])
end
local value = redis.call("GET", KEYS[1])
return {value or false, generation}
`)

// setCacheEntryIfGenerationScript makes writing the value and tracking it for
// invalidation one atomic operation, conditional on no write invalidating the
// generation captured before the backing-store query.
var setCacheEntryIfGenerationScript = redis.NewScript(`
local generation = redis.call("GET", KEYS[3])
if not generation or generation ~= ARGV[1] then
  return 0
end
if ARGV[4] == "1" then
  redis.call("SADD", KEYS[2], KEYS[1])
  -- Track the key before storing a value. If a later command errors, the
  -- index can contain a harmless missing key, but no untracked value exists.
  -- The extra second leaves the index outliving the value if the final
  -- PEXPIRE below errors after SET has already succeeded.
  redis.call("PEXPIRE", KEYS[2], tonumber(ARGV[3]) + 1000)
end
redis.call("SET", KEYS[1], ARGV[2], "PX", ARGV[3])
if ARGV[4] == "1" then
  redis.call("PEXPIRE", KEYS[2], ARGV[3])
end
return 1
`)

// invalidateUserCacheScript serializes cache invalidation with conditional
// refill writes. Removing the generation token is a barrier: stale refills are
// rejected until a later cache read creates a new token.
var invalidateUserCacheScript = redis.NewScript(`
local keys = redis.call("SMEMBERS", KEYS[1])
for _, key in ipairs(keys) do
  redis.call("DEL", key)
end
redis.call("DEL", KEYS[1], KEYS[2])
return #keys
`)

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
// v3 uses a separate value-key namespace from v2 so an older instance cannot
// refill a key that a newer instance will read after invalidation. The user
// index deliberately stays unchanged so invalidation also removes tracked old
// keys. Only the cache identity is normalized; the backing store receives the
// caller's original options and retains its existing retrieval behavior.
func cacheKey(prefix, userID string, opts RetrieveOptions) string {
	return cacheKeyForVersion(prefix, userID, opts, memoryCacheKeyVersion)
}

func cacheKeyForVersion(prefix, userID string, opts RetrieveOptions, version string) string {
	mode := opts.HybridMode
	if !opts.HybridSearch {
		mode = "" // Both stores ignore the fusion mode for vector-only retrieval.
	} else if mode == "" {
		mode = "weighted" // vectorstore.HybridSearchConfig.applyDefaults
	}
	encoded := []byte(version)
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
	return prefix + version + userID + ":" + hash
}

// userIndexKey returns the Redis set key that tracks every cached value key for a
// user. The "u:" namespace is disjoint from the versioned value-key namespace, so an
// index key can never collide with a cached value, regardless of userID content.
func (c *RedisCache) userIndexKey(userID string) string {
	return c.prefix + "u:" + userID
}

func (c *RedisCache) userGenerationKey(userID string) string {
	return c.prefix + "g:" + userID
}

func newCacheGenerationToken() (string, error) {
	var token [16]byte
	if _, err := rand.Read(token[:]); err != nil {
		return "", fmt.Errorf("generate memory cache token: %w", err)
	}
	return hex.EncodeToString(token[:]), nil
}

// Get retrieves cached results for the given options. Returns (nil, nil) on miss or error (no fatal).
func (c *RedisCache) Get(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, bool) {
	results, _, ok := c.GetWithGeneration(ctx, opts)
	return results, ok
}

// GetWithGeneration atomically checks a cached value and returns the user
// generation that a later conditional refill must match. On a miss, retain the
// token while loading from the backing store, then pass it to SetIfGeneration;
// this prevents a completed invalidation from being undone by a slow refill.
// On a hit, return the cached value without refilling it. Redis errors return
// an empty token and disable caching that retrieval result. The token is opaque
// and valid only for a conditional write using the same user and cache options.
func (c *RedisCache) GetWithGeneration(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, string, bool) {
	if c == nil || c.client == nil {
		return nil, "", false
	}
	token, err := newCacheGenerationToken()
	if err != nil {
		logging.Debugf("Memory Redis cache generation token error: %v", err)
		return nil, "", false
	}
	key := cacheKey(c.prefix, opts.UserID, opts)
	result, err := readCacheEntryScript.Run(ctx, c.client,
		[]string{key, c.userGenerationKey(opts.UserID)},
		token, c.ttl.Milliseconds(),
	).Result()
	if err != nil {
		logging.Debugf("Memory Redis cache get error: %v", err)
		return nil, "", false
	}
	parts, ok := result.([]interface{})
	if !ok || len(parts) != 2 {
		logging.Debugf("Memory Redis cache returned an invalid get response")
		return nil, "", false
	}
	generation, ok := redisResultString(parts[1])
	if !ok || generation == "" {
		logging.Debugf("Memory Redis cache returned an invalid generation token")
		return nil, "", false
	}
	if parts[0] == nil {
		return nil, generation, false
	}
	val, ok := redisResultString(parts[0])
	if !ok {
		logging.Warnf("Memory Redis cache returned an invalid value for %s", key)
		return nil, generation, false
	}
	var results []*RetrieveResult
	if err := json.Unmarshal([]byte(val), &results); err != nil {
		logging.Warnf("Memory Redis cache: invalid cached value for %s: %v", key, err)
		return nil, generation, false
	}
	return results, generation, true
}

func redisResultString(value interface{}) (string, bool) {
	switch value := value.(type) {
	case string:
		return value, true
	case []byte:
		return string(value), true
	default:
		return "", false
	}
}

// Set manually prewarms the cache using the current generation.
//
// Do not use Set as the final step of a read-through sequence: if results came
// from a backing-store read begun before an invalidation, acquiring the current
// generation here would allow those old results to be cached. Read-through
// callers must retain the token from GetWithGeneration and use SetIfGeneration.
func (c *RedisCache) Set(ctx context.Context, opts RetrieveOptions, results []*RetrieveResult) {
	if c == nil || c.client == nil {
		return
	}
	_, generation, _ := c.GetWithGeneration(ctx, opts)
	if generation == "" {
		return
	}
	if _, err := c.SetIfGeneration(ctx, opts, results, generation); err != nil {
		logging.Debugf("Memory Redis cache set error: %v", err)
	}
}

// SetIfGeneration stores results only if no successful write has invalidated
// the generation captured by GetWithGeneration before the corresponding
// backing-store read began. Use this with that read's token for read-through
// caching; a false result means an invalidation changed the generation and the
// results were not cached. The token must be used with the same user and cache
// options that produced it.
func (c *RedisCache) SetIfGeneration(ctx context.Context, opts RetrieveOptions, results []*RetrieveResult, generation string) (bool, error) {
	if c == nil || c.client == nil || generation == "" {
		return false, nil
	}
	key := cacheKey(c.prefix, opts.UserID, opts)
	val, err := json.Marshal(results)
	if err != nil {
		logging.Warnf("Memory Redis cache set marshal error: %v", err)
		return false, err
	}
	indexKey := c.userIndexKey(opts.UserID)
	trackForUser := "0"
	if opts.UserID != "" {
		trackForUser = "1"
	}
	stored, err := setCacheEntryIfGenerationScript.Run(ctx, c.client,
		[]string{key, indexKey, c.userGenerationKey(opts.UserID)},
		generation, val, c.ttl.Milliseconds(), trackForUser,
	).Int()
	if err != nil {
		return false, err
	}
	return stored == 1, nil
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
	idxKey := c.userIndexKey(userID)
	if _, err := invalidateUserCacheScript.Run(ctx, c.client,
		[]string{idxKey, c.userGenerationKey(userID)},
	).Int(); err != nil {
		logging.Warnf("Memory Redis cache invalidate failed for user %s; entries may remain cached until TTL: %v", userID, err)
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
