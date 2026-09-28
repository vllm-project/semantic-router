package sessiontools

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// RedisStore is the shared Store implementation for session-scoped sticky
// tool-set selection. It currently targets a standalone Redis endpoint: each
// atomic admission script touches the state, global indexes, quota indexes,
// and generation key, so the key layout is not Redis-Cluster hash-slot safe.
// All values and index members use hashes of the trusted session and quota
// keys; Redis never receives a raw principal or session ID. State payloads use
// State's identity-only JSON representation.
type RedisStore struct {
	client                *redis.Client
	keyPrefix             string
	ttl                   time.Duration
	maxSessions           int
	maxSessionsByIdentity int
	maxStateBytes         int

	lifecycleMu sync.RWMutex
	closed      bool
}

var (
	_ Store                         = (*RedisStore)(nil)
	_ LoadMetadataStore             = (*RedisStore)(nil)
	_ ConditionalDeleteStore        = (*RedisStore)(nil)
	_ RevisionedCompareAndSwapStore = (*RedisStore)(nil)
)

// NewRedisStore constructs a shared store from the resolved
// config.ToolSessionStoreConfig. Construction does not ping Redis: a later
// outage, including one present at startup, is reported by each operation so
// Manager can take its bounded stateless fallback path.
func NewRedisStore(cfg config.ToolSessionStoreConfig) (*RedisStore, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.EffectiveBackend() != config.ToolSessionStoreBackendRedis || cfg.Redis == nil {
		return nil, fmt.Errorf("sessiontools: Redis store requires backend %q", config.ToolSessionStoreBackendRedis)
	}

	timeout := time.Duration(cfg.EffectiveTimeoutMs()) * time.Millisecond
	client := redis.NewClient(&redis.Options{
		Addr:         strings.TrimSpace(cfg.Redis.Address),
		Password:     cfg.Redis.Password,
		DB:           cfg.Redis.Database,
		DialTimeout:  timeout,
		ReadTimeout:  timeout,
		WriteTimeout: timeout,
		PoolTimeout:  timeout,
		MaxRetries:   -1,
	})
	return &RedisStore{
		client:                client,
		keyPrefix:             normalizeRedisKeyPrefix(cfg.EffectiveRedisKeyPrefix()),
		ttl:                   time.Duration(cfg.EffectiveTTLSeconds()) * time.Second,
		maxSessions:           cfg.EffectiveMaxSessions(),
		maxSessionsByIdentity: cfg.EffectiveMaxSessionsByIdentity(),
		maxStateBytes:         cfg.EffectiveMaxStateBytes(),
	}, nil
}

func normalizeRedisKeyPrefix(prefix string) string {
	if strings.HasSuffix(prefix, ":") {
		return prefix
	}
	return prefix + ":"
}

func (s *RedisStore) stateKey(key string) string {
	return s.keyPrefix + "state:" + redisOpaqueKey(key)
}

func (s *RedisStore) quotaIndexKeys(quota QuotaKey) (string, string) {
	digest := redisOpaqueKey(quota.Namespace + "\x00" + quota.Principal)
	base := s.keyPrefix + "identity:" + digest
	return base + ":lru", base + ":expiry"
}

func (s *RedisStore) globalLRUKey() string {
	return s.keyPrefix + "sessions:lru"
}

func (s *RedisStore) globalExpiryKey() string {
	return s.keyPrefix + "sessions:expiry"
}

func (s *RedisStore) generationKey() string {
	return s.keyPrefix + "generation"
}

func (s *RedisStore) revisionKey() string {
	return s.keyPrefix + "revision"
}

func redisOpaqueKey(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])
}

// Load implements Store.
func (s *RedisStore) Load(ctx context.Context, key string) (VersionedState, error) {
	result, _, err := s.LoadWithMetadata(ctx, key)
	return result, err
}

// LoadWithMetadata atomically retrieves one live state without extending its
// lifetime. A successful manager turn refreshes the state and every quota
// index together through CompareAndSwap; failed authorization or validation
// must not keep a session alive merely because it was read.
func (s *RedisStore) LoadWithMetadata(ctx context.Context, key string) (VersionedState, LoadMetadata, error) {
	s.lifecycleMu.RLock()
	defer s.lifecycleMu.RUnlock()
	if s.closed {
		return VersionedState{}, LoadMetadata{}, ErrStoreClosed
	}

	stateKey := s.stateKey(key)
	values, err := s.client.Eval(
		ctx,
		redisLoadScript,
		[]string{stateKey, s.globalLRUKey(), s.globalExpiryKey()},
		durationMilliseconds(s.ttl),
		s.keyPrefix,
	).Slice()
	if err != nil {
		return VersionedState{}, LoadMetadata{}, fmt.Errorf("sessiontools: Redis load failed: %w", err)
	}
	if len(values) == 0 {
		return VersionedState{}, LoadMetadata{}, fmt.Errorf("sessiontools: Redis load returned an empty response")
	}

	status, err := redisReplyString(values[0])
	if err != nil {
		return VersionedState{}, LoadMetadata{}, fmt.Errorf("sessiontools: Redis load returned an invalid status: %w", err)
	}
	switch status {
	case redisLoadStatusMissing:
		return VersionedState{}, LoadMetadata{}, nil
	case redisLoadStatusCorrupted:
		return s.corruptedLoad(ctx, stateKey, values, ErrStateCorrupted)
	case redisLoadStatusExpired:
		metadata, metadataErr := redisLoadMetadata(values)
		if metadataErr != nil {
			return VersionedState{}, LoadMetadata{}, metadataErr
		}
		metadata.Expired = true
		return VersionedState{}, metadata, nil
	case redisLoadStatusFound:
		return s.decodeLoadedState(ctx, stateKey, values)
	default:
		return VersionedState{}, LoadMetadata{}, fmt.Errorf("sessiontools: Redis load returned unknown status %q", status)
	}
}

func (s *RedisStore) decodeLoadedState(
	ctx context.Context,
	stateKey string,
	values []interface{},
) (VersionedState, LoadMetadata, error) {
	if len(values) != redisLoadFoundReplyLength {
		return VersionedState{}, LoadMetadata{}, fmt.Errorf(
			"sessiontools: Redis load returned %d fields for a live state",
			len(values),
		)
	}
	payload, err := redisReplyString(values[1])
	if err != nil {
		return s.corruptedLoad(ctx, stateKey, values, fmt.Errorf("invalid payload: %w", err))
	}
	metadata, err := redisLoadMetadata(values)
	if err != nil {
		return s.corruptedLoad(ctx, stateKey, values, err)
	}
	if metadata.ObservedRevision == 0 || metadata.ObservedGeneration == 0 {
		return s.corruptedLoad(ctx, stateKey, values, fmt.Errorf("revision and generation must be greater than zero"))
	}
	if s.maxStateBytes > 0 && len(payload) > s.maxStateBytes {
		return s.corruptedLoad(ctx, stateKey, values, fmt.Errorf("payload exceeds the configured state size bound"))
	}

	var state State
	if unmarshalErr := json.Unmarshal([]byte(payload), &state); unmarshalErr != nil {
		return s.corruptedLoad(ctx, stateKey, values, fmt.Errorf("invalid state JSON: %w", unmarshalErr))
	}
	if state.Revision != 0 && state.Revision != metadata.ObservedRevision {
		return s.corruptedLoad(ctx, stateKey, values, fmt.Errorf(
			"payload revision %d does not match stored revision %d",
			state.Revision,
			metadata.ObservedRevision,
		))
	}
	state.Revision = metadata.ObservedRevision
	nowMillis, err := redisReplyInt64(values[4])
	if err != nil {
		return s.corruptedLoad(ctx, stateKey, values, fmt.Errorf("invalid last-seen timestamp: %w", err))
	}
	expiresMillis, err := redisReplyInt64(values[5])
	if err != nil {
		return s.corruptedLoad(ctx, stateKey, values, fmt.Errorf("invalid expiry timestamp: %w", err))
	}
	state.LastSeenAt = time.UnixMilli(nowMillis).UTC()
	if state.LastSeenAt.Before(state.CreatedAt) {
		state.LastSeenAt = state.CreatedAt
	}
	state.ExpiresAt = time.UnixMilli(expiresMillis).UTC()
	if err := state.Validate(0, s.maxStateBytes); err != nil {
		return s.corruptedLoad(ctx, stateKey, values, err)
	}
	return VersionedState{State: state.Clone(), Found: true}, metadata, nil
}

func redisLoadMetadata(values []interface{}) (LoadMetadata, error) {
	if len(values) < redisLoadMetadataReplyLength {
		return LoadMetadata{}, fmt.Errorf("sessiontools: Redis load metadata is incomplete")
	}
	revision, err := redisReplyUint64(values[2])
	if err != nil {
		return LoadMetadata{}, fmt.Errorf("sessiontools: Redis load returned an invalid revision: %w", err)
	}
	generation, err := redisReplyUint64(values[3])
	if err != nil {
		return LoadMetadata{}, fmt.Errorf("sessiontools: Redis load returned an invalid generation: %w", err)
	}
	return LoadMetadata{
		ObservedRevision:   revision,
		ObservedGeneration: generation,
	}, nil
}

func (s *RedisStore) corruptedLoad(
	ctx context.Context,
	stateKey string,
	values []interface{},
	cause error,
) (VersionedState, LoadMetadata, error) {
	if len(values) >= redisLoadMetadataReplyLength {
		payload, payloadErr := redisReplyString(values[1])
		revision, revisionErr := redisReplyString(values[2])
		generation, generationErr := redisReplyString(values[3])
		if payloadErr == nil && revisionErr == nil && generationErr == nil {
			_ = s.deleteRawIfCurrent(ctx, stateKey, payload, revision, generation)
		}
	}
	return VersionedState{}, LoadMetadata{}, fmt.Errorf("%w: %w", ErrStateCorrupted, cause)
}

// CompareAndSwap implements Store. Admission, exact global and per-identity
// quota enforcement, LRU replacement, revision comparison, generation
// allocation, and TTL refresh occur in one Redis Lua execution.
func (s *RedisStore) CompareAndSwap(
	ctx context.Context,
	key string,
	expectedRevision uint64,
	next State,
	ttl time.Duration,
	quota QuotaKey,
) (bool, error) {
	_, applied, err := s.CompareAndSwapWithRevision(ctx, key, expectedRevision, next, ttl, quota)
	return applied, err
}

// CompareAndSwapWithRevision implements RevisionedCompareAndSwapStore.
func (s *RedisStore) CompareAndSwapWithRevision(
	ctx context.Context,
	key string,
	expectedRevision uint64,
	next State,
	ttl time.Duration,
	quota QuotaKey,
) (uint64, bool, error) {
	s.lifecycleMu.RLock()
	defer s.lifecycleMu.RUnlock()
	if s.closed {
		return 0, false, ErrStoreClosed
	}
	if ttl <= 0 {
		ttl = s.ttl
	}
	if expectedRevision == ^uint64(0) {
		return 0, false, fmt.Errorf("sessiontools: state revision overflow")
	}

	stored := next.Clone()
	// The Lua script assigns the store-wide revision at the commit point.
	// Zero in the payload marks the hash revision field as authoritative.
	stored.Revision = 0
	now := time.Now().UTC()
	stored.LastSeenAt = now
	stored.ExpiresAt = now.Add(ttl)
	payload, err := json.Marshal(stored)
	if err != nil {
		return 0, false, fmt.Errorf("sessiontools: failed to encode Redis state: %w", err)
	}
	if s.maxStateBytes > 0 && len(payload) > s.maxStateBytes {
		return 0, false, fmt.Errorf(
			"sessiontools: encoded Redis state is %d bytes, exceeds the bound of %d",
			len(payload),
			s.maxStateBytes,
		)
	}

	quotaLRUKey, quotaExpiryKey := s.quotaIndexKeys(quota)
	values, err := s.client.Eval(
		ctx,
		redisCompareAndSwapScript,
		[]string{
			s.stateKey(key),
			s.globalLRUKey(),
			s.globalExpiryKey(),
			quotaLRUKey,
			quotaExpiryKey,
			s.generationKey(),
			s.revisionKey(),
		},
		strconv.FormatUint(expectedRevision, 10),
		string(payload),
		durationMilliseconds(ttl),
		s.maxSessions,
		s.maxSessionsByIdentity,
		s.keyPrefix,
	).Slice()
	if err != nil {
		return 0, false, fmt.Errorf("sessiontools: Redis compare-and-swap failed: %w", err)
	}
	if len(values) == 0 {
		return 0, false, fmt.Errorf("sessiontools: Redis compare-and-swap returned an empty response")
	}
	status, err := redisReplyString(values[0])
	if err != nil {
		return 0, false, fmt.Errorf("sessiontools: Redis compare-and-swap returned an invalid status: %w", err)
	}
	if status == redisCASStatusApplied {
		if len(values) < 2 {
			return 0, false, fmt.Errorf("sessiontools: Redis compare-and-swap omitted the committed revision")
		}
		revision, revisionErr := redisReplyUint64(values[1])
		if revisionErr != nil {
			return 0, false, fmt.Errorf("sessiontools: Redis compare-and-swap returned an invalid revision: %w", revisionErr)
		}
		if revision == 0 {
			return 0, false, fmt.Errorf("sessiontools: Redis compare-and-swap returned a zero revision")
		}
		return revision, true, nil
	}
	if status == redisCASStatusMismatch {
		return 0, false, ErrRevisionMismatch
	}
	if status == redisCASStatusCorrupted {
		return 0, false, ErrStateCorrupted
	}
	return 0, false, fmt.Errorf("sessiontools: Redis compare-and-swap returned unknown status %q", status)
}

// Delete implements Store.
func (s *RedisStore) Delete(ctx context.Context, key string) error {
	s.lifecycleMu.RLock()
	defer s.lifecycleMu.RUnlock()
	if s.closed {
		return ErrStoreClosed
	}
	if err := s.client.Eval(
		ctx,
		redisDeleteScript,
		[]string{s.stateKey(key), s.globalLRUKey(), s.globalExpiryKey()},
		s.keyPrefix,
	).Err(); err != nil {
		return fmt.Errorf("sessiontools: Redis delete failed: %w", err)
	}
	return nil
}

// DeleteIfToken implements ConditionalDeleteStore. Revision and generation
// are compared as decimal strings in Lua so large uint64 revisions never pass
// through Lua's double-precision number representation.
func (s *RedisStore) DeleteIfToken(ctx context.Context, key string, token StateToken) (bool, error) {
	s.lifecycleMu.RLock()
	defer s.lifecycleMu.RUnlock()
	if s.closed {
		return false, ErrStoreClosed
	}
	result, err := s.client.Eval(
		ctx,
		redisDeleteIfTokenScript,
		[]string{s.stateKey(key), s.globalLRUKey(), s.globalExpiryKey()},
		strconv.FormatUint(token.Revision, 10),
		strconv.FormatUint(token.Generation, 10),
		s.keyPrefix,
	).Int64()
	if err != nil {
		return false, fmt.Errorf("sessiontools: Redis conditional delete failed: %w", err)
	}
	return result == 1, nil
}

func (s *RedisStore) deleteRawIfCurrent(
	ctx context.Context,
	stateKey string,
	payload string,
	revision string,
	generation string,
) error {
	return s.client.Eval(
		ctx,
		redisDeleteRawIfCurrentScript,
		[]string{stateKey, s.globalLRUKey(), s.globalExpiryKey()},
		payload,
		revision,
		generation,
		s.keyPrefix,
	).Err()
}

// Close implements Store. It waits for in-flight operations and is
// idempotent; every later operation observes ErrStoreClosed.
func (s *RedisStore) Close() error {
	s.lifecycleMu.Lock()
	defer s.lifecycleMu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	return s.client.Close()
}

func durationMilliseconds(duration time.Duration) int64 {
	milliseconds := duration.Milliseconds()
	if milliseconds < 1 {
		return 1
	}
	return milliseconds
}

func redisReplyString(value interface{}) (string, error) {
	switch typed := value.(type) {
	case string:
		return typed, nil
	case []byte:
		return string(typed), nil
	case int64:
		return strconv.FormatInt(typed, 10), nil
	default:
		return "", fmt.Errorf("unexpected Redis reply type %T", value)
	}
}

func redisReplyUint64(value interface{}) (uint64, error) {
	encoded, err := redisReplyString(value)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(encoded, 10, 64)
}

func redisReplyInt64(value interface{}) (int64, error) {
	encoded, err := redisReplyString(value)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(encoded, 10, 64)
}
