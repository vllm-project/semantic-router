package contextcompression

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

var ErrRecoveryNotFound = errors.New("compression recovery entry not found")

type RedisRecoveryStoreOptions struct {
	Address       string
	Password      string
	Database      int
	TLSConfig     *tls.Config
	KeyPrefix     string
	MaxTotalBytes int
}

type RedisRecoveryStore struct {
	client        *redis.Client
	prefix        string
	maxTotalBytes int
}

func NewRedisRecoveryStore(options RedisRecoveryStoreOptions) (*RedisRecoveryStore, error) {
	if options.Address == "" {
		return nil, fmt.Errorf("recovery Redis address is required")
	}
	if options.KeyPrefix == "" {
		options.KeyPrefix = "vsr:context-recovery:v1:"
	}
	if options.MaxTotalBytes <= 0 {
		options.MaxTotalBytes = 256 * 1024 * 1024
	}
	store := &RedisRecoveryStore{
		client: redis.NewClient(&redis.Options{
			Addr:      options.Address,
			Password:  options.Password,
			DB:        options.Database,
			TLSConfig: options.TLSConfig,
		}),
		prefix:        options.KeyPrefix,
		maxTotalBytes: options.MaxTotalBytes,
	}
	return store, nil
}

func (store *RedisRecoveryStore) Put(
	ctx context.Context,
	entry RecoveryEntry,
	ttl time.Duration,
) error {
	if store == nil || store.client == nil {
		return fmt.Errorf("recovery Redis store is unavailable")
	}
	if ttl <= 0 {
		return fmt.Errorf("recovery TTL must be positive")
	}
	payload, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("encode recovery entry: %w", err)
	}
	key := store.entryKey(entry.Scope, entry.Key)
	metadataTTL := int64((ttl + time.Hour) / time.Second)
	result, err := store.client.Eval(
		ctx,
		recoveryPutScript,
		[]string{
			key,
			store.prefix + "total_bytes",
			store.prefix + "sizes",
			store.prefix + "expiries",
			store.scopeIndexKey(entry.Scope),
		},
		string(payload),
		len(payload),
		store.maxTotalBytes,
		int64(ttl/time.Second),
		time.Now().Add(ttl).UnixMilli(),
		metadataTTL,
		time.Now().UnixMilli(),
	).Int64()
	if err != nil {
		return fmt.Errorf("write recovery entry: %w", err)
	}
	if result == 0 {
		return fmt.Errorf("recovery store byte limit exceeded")
	}
	return nil
}

func (store *RedisRecoveryStore) InvalidateScope(
	ctx context.Context,
	scope string,
) (int64, error) {
	if store == nil || store.client == nil {
		return 0, fmt.Errorf("recovery Redis store is unavailable")
	}
	deleted, err := store.client.Eval(
		ctx,
		recoveryInvalidateScopeScript,
		[]string{
			store.prefix + "total_bytes",
			store.prefix + "sizes",
			store.prefix + "expiries",
			store.scopeIndexKey(scope),
		},
	).Int64()
	if err != nil {
		return 0, fmt.Errorf("invalidate recovery scope: %w", err)
	}
	return deleted, nil
}

func (store *RedisRecoveryStore) Get(
	ctx context.Context,
	scope string,
	key string,
) (RecoveryEntry, error) {
	if store == nil || store.client == nil {
		return RecoveryEntry{}, fmt.Errorf("recovery Redis store is unavailable")
	}
	payload, err := store.client.Get(ctx, store.entryKey(scope, key)).Bytes()
	if errors.Is(err, redis.Nil) {
		return RecoveryEntry{}, ErrRecoveryNotFound
	}
	if err != nil {
		return RecoveryEntry{}, fmt.Errorf("read recovery entry: %w", err)
	}
	var entry RecoveryEntry
	if err := json.Unmarshal(payload, &entry); err != nil {
		return RecoveryEntry{}, fmt.Errorf("decode recovery entry: %w", err)
	}
	if entry.Scope != scope || entry.Key != key {
		return RecoveryEntry{}, ErrRecoveryNotFound
	}
	if !entry.ExpiresAt.IsZero() && time.Now().After(entry.ExpiresAt) {
		return RecoveryEntry{}, ErrRecoveryNotFound
	}
	return entry, nil
}

func (store *RedisRecoveryStore) Health(ctx context.Context) error {
	if store == nil || store.client == nil {
		return fmt.Errorf("recovery Redis store is unavailable")
	}
	return store.client.Ping(ctx).Err()
}

func (store *RedisRecoveryStore) Close() error {
	if store == nil || store.client == nil {
		return nil
	}
	return store.client.Close()
}

func (store *RedisRecoveryStore) entryKey(scope string, key string) string {
	return store.prefix + "entry:" + recoveryKey(scope, key)
}

func (store *RedisRecoveryStore) scopeIndexKey(scope string) string {
	return store.prefix + "scope:" + recoveryKey("scope", scope)
}

const recoveryPutScript = `
local expired = redis.call("ZRANGEBYSCORE", KEYS[4], "-inf", ARGV[7])
for _, member in ipairs(expired) do
  local size = tonumber(redis.call("HGET", KEYS[3], member) or "0")
  redis.call("HDEL", KEYS[3], member)
  redis.call("ZREM", KEYS[4], member)
  if size > 0 then
    redis.call("DECRBY", KEYS[2], size)
  end
end
local old_size = tonumber(redis.call("HGET", KEYS[3], KEYS[1]) or "0")
local current = tonumber(redis.call("GET", KEYS[2]) or "0")
local next_total = current - old_size + tonumber(ARGV[2])
if next_total > tonumber(ARGV[3]) then
  return 0
end
redis.call("SET", KEYS[1], ARGV[1], "EX", ARGV[4])
redis.call("SET", KEYS[2], next_total, "EX", ARGV[6])
redis.call("HSET", KEYS[3], KEYS[1], ARGV[2])
redis.call("EXPIRE", KEYS[3], ARGV[6])
redis.call("ZADD", KEYS[4], ARGV[5], KEYS[1])
redis.call("EXPIRE", KEYS[4], ARGV[6])
redis.call("SADD", KEYS[5], KEYS[1])
redis.call("EXPIRE", KEYS[5], ARGV[6])
return 1
`

const recoveryInvalidateScopeScript = `
local members = redis.call("SMEMBERS", KEYS[4])
local removed_bytes = 0
for _, member in ipairs(members) do
  local size = tonumber(redis.call("HGET", KEYS[2], member) or "0")
  removed_bytes = removed_bytes + size
  redis.call("DEL", member)
  redis.call("HDEL", KEYS[2], member)
  redis.call("ZREM", KEYS[3], member)
end
if removed_bytes > 0 then
  local current = tonumber(redis.call("GET", KEYS[1]) or "0")
  redis.call("SET", KEYS[1], math.max(0, current - removed_bytes))
end
redis.call("DEL", KEYS[4])
return #members
`
