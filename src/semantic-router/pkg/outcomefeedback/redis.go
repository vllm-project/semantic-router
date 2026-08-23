package outcomefeedback

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	defaultAbuseLimit  = 120
	defaultAbuseWindow = time.Minute
)

var outcomeAbuseScript = redis.NewScript(`
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
local ttl = redis.call('PTTL', KEYS[1])
if ttl < 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
  ttl = tonumber(ARGV[2])
end
if current > tonumber(ARGV[1]) then
  return {0, ttl}
end
return {1, ttl}
`)

var publishProjectionScript = redis.NewScript(`
local current_revision = redis.call('HGET', KEYS[1], 'revision')
if current_revision then
  current_revision = tonumber(current_revision)
  local next_revision = tonumber(ARGV[1])
  if current_revision > next_revision then
    return 2
  end
  if current_revision == next_revision then
    if redis.call('HGET', KEYS[1], 'digest') == ARGV[2] then
      return 1
    end
    return -1
  end
end
redis.call('HSET', KEYS[1],
  'schema', ARGV[4],
  'revision', ARGV[1],
  'digest', ARGV[2],
  'payload', ARGV[3])
return 1
`)

type RedisAbuseLimiterOptions struct {
	Client    redis.Scripter
	KeyPrefix string
	Limit     int64
	Window    time.Duration
}

// RedisAbuseLimiter is a deliberately separate global budget for public
// feedback ingestion. Its keys and Lua mutation never touch inference request,
// token, concurrency, or cost meters.
type RedisAbuseLimiter struct {
	client    redis.Scripter
	keyPrefix string
	limit     int64
	window    time.Duration
}

var _ AbuseLimiter = (*RedisAbuseLimiter)(nil)

func NewRedisAbuseLimiter(options RedisAbuseLimiterOptions) (*RedisAbuseLimiter, error) {
	if options.Client == nil {
		return nil, errors.New("outcome abuse-limit client is required")
	}
	prefix, err := validateKeyPrefix(options.KeyPrefix)
	if err != nil {
		return nil, err
	}
	limit := options.Limit
	if limit == 0 {
		limit = defaultAbuseLimit
	}
	if limit < 1 || limit > 1_000_000 {
		return nil, errors.New("outcome abuse limit must be between 1 and 1000000")
	}
	window := options.Window
	if window == 0 {
		window = defaultAbuseWindow
	}
	if window < time.Second || window > 24*time.Hour {
		return nil, errors.New("outcome abuse window must be between one second and 24 hours")
	}
	return &RedisAbuseLimiter{client: options.Client, keyPrefix: prefix, limit: limit, window: window}, nil
}

func (limiter *RedisAbuseLimiter) Allow(ctx context.Context, caller Caller) (AbuseDecision, error) {
	if limiter == nil || limiter.client == nil {
		return AbuseDecision{}, ErrUnavailable
	}
	if err := caller.Validate(); err != nil {
		return AbuseDecision{}, err
	}
	result, err := outcomeAbuseScript.Run(ctx, limiter.client, []string{limiter.key(caller)},
		limiter.limit, limiter.window.Milliseconds()).Int64Slice()
	if err != nil || len(result) != 2 {
		return AbuseDecision{}, fmt.Errorf("%w: mutate global outcome abuse budget", ErrUnavailable)
	}
	retry := time.Duration(result[1]) * time.Millisecond
	if retry < 0 {
		retry = limiter.window
	}
	return AbuseDecision{Allowed: result[0] == 1, RetryAfter: retry}, nil
}

func (limiter *RedisAbuseLimiter) key(caller Caller) string {
	digest := sha256.Sum256([]byte(caller.APIKeyID))
	return fmt.Sprintf("%s:outcome:abuse:{%s}:%s", limiter.keyPrefix, caller.NamespaceID,
		hex.EncodeToString(digest[:16]))
}

type RedisProjectionStoreOptions struct {
	Client    redis.UniversalClient
	KeyPrefix string
}

// RedisProjectionStore publishes and reads only revisioned, rebuildable
// learning projections. PostgreSQL remains the durable source of truth.
type RedisProjectionStore struct {
	client    redis.UniversalClient
	keyPrefix string
}

var _ ProjectionPublisher = (*RedisProjectionStore)(nil)

func NewRedisProjectionStore(options RedisProjectionStoreOptions) (*RedisProjectionStore, error) {
	if options.Client == nil {
		return nil, errors.New("outcome projection client is required")
	}
	prefix, err := validateKeyPrefix(options.KeyPrefix)
	if err != nil {
		return nil, err
	}
	return &RedisProjectionStore{client: options.Client, keyPrefix: prefix}, nil
}

func (store *RedisProjectionStore) Publish(
	ctx context.Context,
	projection Projection,
	payload []byte,
	digest [sha256.Size]byte,
) error {
	if store == nil || store.client == nil {
		return ErrUnavailable
	}
	canonical, canonicalDigest, err := projection.Canonical()
	if err != nil {
		return err
	}
	if string(canonical) != string(payload) || canonicalDigest != digest {
		return fmt.Errorf("%w: projection payload does not match its identity", ErrInvalid)
	}
	result, err := publishProjectionScript.Run(ctx, store.client, []string{store.key(projection.NamespaceID)},
		projection.Revision, DigestHex(digest), payload, ProjectionSchema).Int64()
	if err != nil {
		return fmt.Errorf("%w: publish outcome learning projection", ErrUnavailable)
	}
	if result == -1 {
		return fmt.Errorf("%w: outcome projection revision has another digest", ErrUnavailable)
	}
	if result != 1 && result != 2 {
		return fmt.Errorf("%w: outcome projection publication returned an invalid result", ErrUnavailable)
	}
	return nil
}

func (store *RedisProjectionStore) Read(ctx context.Context, namespaceID string) (Projection, error) {
	if store == nil || store.client == nil || !canonicalIdentifier(namespaceID, MaximumReplayIDSize) {
		return Projection{}, ErrInvalid
	}
	values, err := store.client.HGetAll(ctx, store.key(namespaceID)).Result()
	if err != nil {
		return Projection{}, fmt.Errorf("%w: read outcome learning projection", ErrUnavailable)
	}
	if len(values) == 0 {
		return Projection{}, ErrNotFound
	}
	if values["schema"] != ProjectionSchema || values["payload"] == "" || values["digest"] == "" {
		return Projection{}, fmt.Errorf("%w: outcome learning projection is corrupt", ErrUnavailable)
	}
	projection, err := decodeProjection([]byte(values["payload"]), values["digest"])
	if err != nil {
		return Projection{}, fmt.Errorf("%w: outcome learning projection is corrupt", ErrUnavailable)
	}
	revision, err := strconv.ParseInt(values["revision"], 10, 64)
	if err != nil || revision != projection.Revision || projection.NamespaceID != namespaceID {
		return Projection{}, fmt.Errorf("%w: outcome learning projection identity is corrupt", ErrUnavailable)
	}
	return projection, nil
}

func (store *RedisProjectionStore) key(namespaceID string) string {
	return fmt.Sprintf("%s:outcome:projection:{%s}:active", store.keyPrefix, namespaceID)
}

func validateKeyPrefix(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > 128 || strings.ContainsAny(value, "{}\x00\r\n\t ") {
		return "", errors.New("outcome Redis key prefix is invalid")
	}
	return strings.TrimSuffix(value, ":"), nil
}
