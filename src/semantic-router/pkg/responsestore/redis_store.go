package responsestore

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
	"sigs.k8s.io/yaml"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// RedisStore implements the CombinedStore interface using Redis as the backend.
// It supports both standalone Redis and Redis Cluster deployments.
type RedisStore struct {
	client    redis.UniversalClient // Works with both standalone and cluster
	config    RedisStoreConfig
	keyPrefix string
	ttl       time.Duration
	enabled   bool

	// indexResponseOverride, when non-nil, replaces indexResponse's Redis
	// calls entirely. Unexported test-only seam: a real Redis outage would
	// fail the payload SETNX and the index write together, which cannot
	// isolate "payload write succeeded, index write failed" for rollback and
	// repair tests. Never set outside _test.go files in this package.
	indexResponseOverride func(ctx context.Context, conversationID, responseID string, createdAt int64) error

	// scanInvocations counts calls to scanResponsePayloads. Unexported
	// test-only seam proving the O(N) legacy scan runs at most once per
	// conversation per empty-marker/index lifetime, not on every read of an
	// empty or unknown conversation. Not read anywhere outside _test.go.
	scanInvocations int

	// lazyBackfillPreScanHook, when non-nil, runs once at the top of
	// lazyBackfillConversationIndex before the scan starts. Unexported
	// test-only seam for deterministically landing a concurrent indexed
	// write in the window the blueprint calls out (§2.2 Redis concurrency):
	// a writer's SETNX + index ZADD completing while a lazy scan for the
	// same conversation is in flight must not be undone by the scan.
	lazyBackfillPreScanHook func()
}

const (
	// ResponseKeyPrefix for response keys
	// Combined with key_prefix (default "sr:"): sr:response:resp_xxxxx
	ResponseKeyPrefix = "response:"

	// ConversationKeyPrefix for conversation keys
	// Combined with key_prefix (default "sr:"): sr:conversation:conv_xxxxx
	ConversationKeyPrefix = "conversation:"

	// ConversationIndexKeyPrefix for the sorted set of a conversation's response
	// IDs, scored by created_at: sr:conversation-index:conv_xxxxx
	//
	// Must not start with ConversationKeyPrefix, or the sr:conversation:* scan in
	// ListConversations would read these sorted sets as conversation JSON.
	ConversationIndexKeyPrefix = "conversation-index:"

	// ConversationIndexEmptyMarkerKeyPrefix marks a conversation ID for which a
	// completed lazy legacy scan found no live response payloads to index:
	// sr:conversation-index-empty:conv_xxxxx, value "v1".
	//
	// Prevents a caller from forcing repeated full keyspace scans by repeatedly
	// listing the same empty or unknown conversation. Must not start with
	// ConversationKeyPrefix or ResponseKeyPrefix, for the same scan-isolation
	// reason as ConversationIndexKeyPrefix.
	ConversationIndexEmptyMarkerKeyPrefix = "conversation-index-empty:"

	// ConversationIndexLockKeyPrefix guards the one-time lazy legacy scan for a
	// conversation against a stampede of concurrent readers all missing the
	// index at once: sr:conversation-index-lock:conv_xxxxx, TTL-bound token.
	//
	// An optimization against duplicate scan work, not a correctness
	// dependency: a reader that fails to acquire it still gets correct results,
	// see ensureConversationIndex. Same scan-isolation constraint as the marker
	// and index key prefixes.
	ConversationIndexLockKeyPrefix = "conversation-index-lock:"

	// conversationIndexMigrationLockTTL bounds how long a single reader holds
	// the lazy-scan migration lock before it is considered abandoned.
	conversationIndexMigrationLockTTL = 30 * time.Second

	// redisScanCount is the SCAN COUNT hint used when walking the response
	// keyspace for lazy legacy backfill. A hint, not a hard limit — Redis may
	// return more or fewer keys per cursor step.
	redisScanCount = 1000

	// redisBackfillBatchSize bounds how many keys are GET-pipelined, and how
	// many discovered members are ZADD-ed, per round trip during lazy legacy
	// backfill — keeps a single conversation's backfill from building one
	// unbounded pipeline or command.
	redisBackfillBatchSize = 256
)

// compareDeleteScript deletes KEYS[1] only if its current value equals
// ARGV[1]. Single-key: legal in Redis Cluster with no hash-tagging needed.
// Used to roll back a response payload after its index write fails, without
// risking a blind DEL of a value a concurrent writer stored in the same slot
// after the original payload's TTL expired.
var compareDeleteScript = redis.NewScript(`
if redis.call("GET", KEYS[1]) == ARGV[1] then
	return redis.call("DEL", KEYS[1])
end
return 0
`)

// The function validates configuration, establishes connection, and tests connectivity.
func NewRedisStore(config StoreConfig) (*RedisStore, error) {
	logging.ComponentEvent("responsestore", "redis_store_init_started", map[string]interface{}{
		"cluster_mode":    config.Redis.ClusterMode,
		"external_config": config.Redis.ConfigPath != "",
		"ttl_seconds":     config.TTLSeconds,
	})

	ttl := DefaultTTL
	if config.TTLSeconds > 0 {
		ttl = time.Duration(config.TTLSeconds) * time.Second
	}

	finalCfg, err := loadRedisStoreConfig(config.Redis)
	if err != nil {
		return nil, fmt.Errorf("failed to load Redis config: %w", err)
	}

	if validateErr := validateRedisConfig(finalCfg); validateErr != nil {
		return nil, fmt.Errorf("invalid Redis config: %w", validateErr)
	}

	applyRedisConfigDefaults(&finalCfg)

	keyPrefix := finalCfg.KeyPrefix
	if !strings.HasSuffix(keyPrefix, ":") {
		keyPrefix += ":"
	}

	// Create Redis client (standalone or cluster)
	client, err := createRedisClient(finalCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create Redis client: %w", err)
	}

	store := &RedisStore{
		client:    client,
		config:    finalCfg,
		keyPrefix: keyPrefix,
		ttl:       ttl,
		enabled:   true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := store.CheckConnection(ctx); err != nil {
		client.Close()
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	mode := "standalone"
	if finalCfg.ClusterMode {
		mode = "cluster"
	}
	logging.ComponentEvent("responsestore", "redis_store_initialized", map[string]interface{}{
		"mode":         mode,
		"cluster_mode": finalCfg.ClusterMode,
		"key_prefix":   keyPrefix,
		"ttl_seconds":  ttl.Seconds(),
		"pool_size":    finalCfg.PoolSize,
	})

	return store, nil
}

func loadRedisStoreConfig(cfg RedisStoreConfig) (RedisStoreConfig, error) {
	// If no external config, return inline config as-is
	if cfg.ConfigPath == "" {
		logging.ComponentDebugEvent("responsestore", "redis_store_config_source_selected", map[string]interface{}{
			"source": "inline",
		})
		return cfg, nil
	}

	// Load external configuration
	logging.ComponentDebugEvent("responsestore", "redis_store_config_source_selected", map[string]interface{}{
		"source":      "file",
		"config_path": cfg.ConfigPath,
	})

	data, err := os.ReadFile(cfg.ConfigPath)
	if err != nil {
		return cfg, fmt.Errorf("failed to read config file %s: %w", cfg.ConfigPath, err)
	}

	var fileCfg RedisStoreConfig
	if err := yaml.Unmarshal(data, &fileCfg); err != nil {
		return cfg, fmt.Errorf("failed to parse config file %s: %w", cfg.ConfigPath, err)
	}

	logging.ComponentDebugEvent("responsestore", "redis_store_config_loaded", map[string]interface{}{
		"cluster_mode": fileCfg.ClusterMode,
		"address":      fileCfg.Address,
		"nodes":        len(fileCfg.ClusterAddresses),
	})

	// External file takes precedence
	return fileCfg, nil
}

func validateRedisConfig(cfg RedisStoreConfig) error {
	// Cluster mode validation
	if cfg.ClusterMode {
		// Cluster requires ClusterAddresses
		if len(cfg.ClusterAddresses) == 0 {
			return fmt.Errorf("cluster_mode is true but cluster_addresses is empty")
		}
		// Cluster only supports DB 0
		if cfg.DB != 0 {
			return fmt.Errorf("redis cluster only supports db 0, got db: %d", cfg.DB)
		}
	} else if cfg.Address == "" {
		// Standalone requires Address
		return fmt.Errorf("address is required for standalone Redis")
	}

	// DB range validation (0-15 for standalone)
	if cfg.DB < 0 || cfg.DB > 15 {
		return fmt.Errorf("invalid DB number %d (must be 0-15)", cfg.DB)
	}

	// TLS validation
	if cfg.TLSEnabled {
		if cfg.TLSCertPath == "" || cfg.TLSKeyPath == "" {
			return fmt.Errorf("tls_cert_path and tls_key_path are required when TLS is enabled")
		}
		// Check if cert files exist
		if _, err := os.Stat(cfg.TLSCertPath); os.IsNotExist(err) {
			return fmt.Errorf("TLS cert file not found: %s", cfg.TLSCertPath)
		}
		if _, err := os.Stat(cfg.TLSKeyPath); os.IsNotExist(err) {
			return fmt.Errorf("TLS key file not found: %s", cfg.TLSKeyPath)
		}
	}

	return nil
}

func applyRedisConfigDefaults(cfg *RedisStoreConfig) {
	if cfg.KeyPrefix == "" {
		cfg.KeyPrefix = "sr:" // Base prefix only, types are added by constants
	}
	if cfg.PoolSize == 0 {
		cfg.PoolSize = 10
	}
	if cfg.MinIdleConns == 0 {
		cfg.MinIdleConns = 2
	}
	if cfg.MaxRetries == 0 {
		cfg.MaxRetries = 3
	}
	if cfg.DialTimeout == 0 {
		cfg.DialTimeout = 5
	}
	if cfg.ReadTimeout == 0 {
		cfg.ReadTimeout = 3
	}
	if cfg.WriteTimeout == 0 {
		cfg.WriteTimeout = 3
	}
}

// createRedisClient creates a Redis client (standalone or cluster) based on configuration.
func createRedisClient(cfg RedisStoreConfig) (redis.UniversalClient, error) {
	// Build TLS config if enabled
	var tlsConfig *tls.Config
	if cfg.TLSEnabled {
		cert, err := tls.LoadX509KeyPair(cfg.TLSCertPath, cfg.TLSKeyPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load TLS certificate: %w", err)
		}

		tlsConfig = &tls.Config{
			Certificates: []tls.Certificate{cert},
		}

		// Load CA certificate if provided
		if cfg.TLSCAPath != "" {
			caCert, err := os.ReadFile(cfg.TLSCAPath)
			if err != nil {
				return nil, fmt.Errorf("failed to read CA certificate: %w", err)
			}
			caCertPool := x509.NewCertPool()
			if !caCertPool.AppendCertsFromPEM(caCert) {
				return nil, fmt.Errorf("failed to parse CA certificate")
			}
			tlsConfig.RootCAs = caCertPool
		}

		logging.ComponentDebugEvent("responsestore", "redis_store_tls_enabled", map[string]interface{}{
			"ca_configured": cfg.TLSCAPath != "",
		})
	}

	// Create client based on mode
	if cfg.ClusterMode {
		logging.ComponentDebugEvent("responsestore", "redis_client_create_started", map[string]interface{}{
			"mode":      "cluster",
			"nodes":     len(cfg.ClusterAddresses),
			"pool_size": cfg.PoolSize,
		})

		return redis.NewClusterClient(&redis.ClusterOptions{
			Addrs:        cfg.ClusterAddresses,
			Password:     cfg.Password,
			PoolSize:     cfg.PoolSize,
			MinIdleConns: cfg.MinIdleConns,
			MaxRetries:   cfg.MaxRetries,
			DialTimeout:  time.Duration(cfg.DialTimeout) * time.Second,
			ReadTimeout:  time.Duration(cfg.ReadTimeout) * time.Second,
			WriteTimeout: time.Duration(cfg.WriteTimeout) * time.Second,
			TLSConfig:    tlsConfig,
		}), nil
	}

	// Standalone mode
	logging.ComponentDebugEvent("responsestore", "redis_client_create_started", map[string]interface{}{
		"mode":      "standalone",
		"address":   cfg.Address,
		"db":        cfg.DB,
		"pool_size": cfg.PoolSize,
	})

	return redis.NewClient(&redis.Options{
		Addr:         cfg.Address,
		Password:     cfg.Password,
		DB:           cfg.DB,
		PoolSize:     cfg.PoolSize,
		MinIdleConns: cfg.MinIdleConns,
		MaxRetries:   cfg.MaxRetries,
		DialTimeout:  time.Duration(cfg.DialTimeout) * time.Second,
		ReadTimeout:  time.Duration(cfg.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.WriteTimeout) * time.Second,
		TLSConfig:    tlsConfig,
	}), nil
}

// buildKey constructs a Redis key with the proper prefix.
func (s *RedisStore) buildKey(suffix string) string {
	return s.keyPrefix + suffix
}

// conversationIndexKey returns the sorted set indexing a conversation's responses.
func (s *RedisStore) conversationIndexKey(conversationID string) string {
	return s.buildKey(ConversationIndexKeyPrefix + conversationID)
}

// emptyConversationIndexMarkerKey returns the marker set after a completed
// lazy legacy scan finds no live response payloads for a conversation. Its
// presence means "already scanned, confirmed empty or unknown" — checked
// only after the index key itself is confirmed absent, so a stale marker can
// never hide an index created concurrently.
func (s *RedisStore) emptyConversationIndexMarkerKey(conversationID string) string {
	return s.buildKey(ConversationIndexEmptyMarkerKeyPrefix + conversationID)
}

// conversationIndexLockKey returns the short-lived lock guarding a
// conversation's lazy legacy scan against concurrent duplicate scans.
func (s *RedisStore) conversationIndexLockKey(conversationID string) string {
	return s.buildKey(ConversationIndexLockKeyPrefix + conversationID)
}

func (s *RedisStore) CheckConnection(ctx context.Context) error {
	if !s.enabled {
		return fmt.Errorf("redis store is disabled")
	}

	// Use PING command to test connection
	if err := s.client.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("redis ping failed: %w", err)
	}

	logging.Debugf("RedisStore: connection check passed")
	return nil
}

func (s *RedisStore) Close() error {
	if s.client != nil {
		logging.Infof("RedisStore: closing connection")
		return s.client.Close()
	}
	return nil
}

func (s *RedisStore) IsEnabled() bool {
	return s.enabled
}

// Response Store Methods

func (s *RedisStore) StoreResponse(ctx context.Context, response *responseapi.StoredResponse) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if response == nil || response.ID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + response.ID)

	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	// Atomic existence check, and it lands the payload before the index entry, so
	// a member always postdates its payload — what makes prune-on-missing safe.
	stored, err := s.client.SetNX(ctx, key, data, s.ttl).Result()
	if err != nil {
		return fmt.Errorf("failed to store response in Redis: %w", err)
	}
	if !stored {
		// The payload already here is the source of truth for whether this
		// retry needs an index repair, not the caller's attempted request:
		// verified inside repairExistingResponseIndex.
		if response.ConversationID != "" {
			if repairErr := s.repairExistingResponseIndex(ctx, response); repairErr != nil {
				return repairErr
			}
		}
		return ErrAlreadyExists
	}

	if response.ConversationID == "" {
		return nil
	}

	if err := s.indexResponse(ctx, response.ConversationID, response.ID, response.CreatedAt); err != nil {
		indexErr := fmt.Errorf("failed to index response in Redis: %w", err)

		// Compare-delete rollback: never a blind DEL. Only removes the
		// payload if it is still exactly what this call wrote, so a
		// concurrent writer that stored a new value after this payload's TTL
		// expired is never clobbered (the ABA race the blueprint calls out).
		deleted, rollbackErr := s.compareDeleteResponsePayload(ctx, key, data)
		if rollbackErr != nil {
			return fmt.Errorf("%w (rollback failed: %v)", indexErr, rollbackErr)
		}
		if !deleted {
			return fmt.Errorf("%w (payload changed before rollback, left in place)", indexErr)
		}
		return indexErr
	}

	return nil
}

// compareDeleteResponsePayload deletes key only if its current value equals
// expected. Used to roll back a response payload after its index write
// fails, without risking a blind DEL removing a value a concurrent writer
// stored after the original payload expired on its TTL; also reused by
// ensureConversationIndex to release the migration lock without releasing
// one it doesn't hold (e.g. one that expired and was re-acquired).
//
// Single-key Lua script — touches only KEYS[1] — so it stays legal in Redis
// Cluster; a two-key script spanning the response and index keys would not.
func (s *RedisStore) compareDeleteResponsePayload(ctx context.Context, key string, expected []byte) (bool, error) {
	res, err := compareDeleteScript.Run(ctx, s.client, []string{key}, expected).Result()
	if err != nil {
		return false, fmt.Errorf("failed to compare-delete response payload %s: %w", key, err)
	}

	deleted, ok := res.(int64)
	if !ok {
		return false, fmt.Errorf("unexpected compare-delete result type %T for %s", res, key)
	}

	return deleted > 0, nil
}

// repairExistingResponseIndex runs when StoreResponse's SETNX finds the
// response ID already stored. It never trusts the caller's attempted
// payload: the stored response is read back and is the only source of truth
// for whether — and under which conversation — the index should be
// repaired. A duplicate ID whose stored payload belongs to a different
// conversation than the one the caller attempted must not poison that
// conversation's index.
func (s *RedisStore) repairExistingResponseIndex(ctx context.Context, attempted *responseapi.StoredResponse) error {
	stored, err := s.GetResponse(ctx, attempted.ID)
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			// SETNX reported existence, but the payload is gone now (raced
			// with a delete, or expired). Nothing to repair from; keep the
			// duplicate contract.
			return nil
		}
		return fmt.Errorf("failed to read stored response %s for index repair: %w", attempted.ID, err)
	}

	if stored.ConversationID == "" || stored.ConversationID != attempted.ConversationID {
		// Either no index is expected, or the stored payload proves this
		// duplicate belongs to a different conversation than attempted.
		// Repairing the attempted conversation's index here would be
		// indexing a response that conversation does not actually own.
		return nil
	}

	if err := s.indexResponse(ctx, stored.ConversationID, stored.ID, stored.CreatedAt); err != nil {
		return fmt.Errorf("response already exists but failed to repair conversation index: %w", err)
	}

	return nil
}

func (s *RedisStore) GetResponse(ctx context.Context, responseID string) (*responseapi.StoredResponse, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if responseID == "" {
		return nil, ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + responseID)

	data, err := s.client.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to get response from Redis: %w", err)
	}

	var response responseapi.StoredResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, fmt.Errorf("failed to deserialize response: %w", err)
	}

	return &response, nil
}

func (s *RedisStore) UpdateResponse(ctx context.Context, response *responseapi.StoredResponse) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if response == nil || response.ID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + response.ID)

	// Doubles as the existence check, and detects a move between conversations.
	previousConversationID, err := s.storedConversationID(ctx, response.ID)
	if err != nil {
		return err
	}

	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to update response in Redis: %w", err)
	}

	// TODO(#2814 Phase 5): make repairable — restore previous payload on index failure.
	if previousConversationID != "" && previousConversationID != response.ConversationID {
		if err := s.unindexResponse(ctx, previousConversationID, response.ID); err != nil {
			logging.Warnf("RedisStore: failed to remove response %s from previous conversation %s index: %v",
				response.ID, previousConversationID, err)
		}
	}
	if err := s.indexResponse(ctx, response.ConversationID, response.ID, response.CreatedAt); err != nil {
		logging.Warnf("RedisStore: failed to index response %s in conversation %s: %v",
			response.ID, response.ConversationID, err)
	}

	return nil
}

func (s *RedisStore) DeleteResponse(ctx context.Context, responseID string) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if responseID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ResponseKeyPrefix + responseID)

	// Needed to drop the index entry; also the existence check.
	conversationID, err := s.storedConversationID(ctx, responseID)
	if err != nil {
		return err
	}

	deleted, err := s.client.Del(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to delete response from Redis: %w", err)
	}
	if deleted == 0 {
		return ErrNotFound
	}

	// Best-effort: the payload delete above is the user-visible operation, and
	// a stale index entry is pruned by the next listing that finds it missing.
	if err := s.unindexResponse(ctx, conversationID, responseID); err != nil {
		logging.Warnf("RedisStore: failed to remove response %s from conversation %s index: %v",
			responseID, conversationID, err)
	}

	return nil
}

// GetConversationChain retrieves the full conversation chain for a response.
// It follows the previous_response_id links backwards to build the complete history.
func (s *RedisStore) GetConversationChain(ctx context.Context, responseID string) ([]*responseapi.StoredResponse, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if responseID == "" {
		return nil, ErrInvalidInput
	}

	// Phase 1: Collect response IDs by following the chain
	responseIDs, err := s.collectChainIDs(ctx, responseID)
	if err != nil {
		return nil, err
	}

	if len(responseIDs) == 0 {
		return []*responseapi.StoredResponse{}, nil
	}

	// Phase 2: Fetch all responses using pipelining
	chain, _, err := s.fetchResponsesPipelined(ctx, responseIDs)
	if err != nil {
		return nil, err
	}

	// Phase 3: Reverse chain to get chronological order (oldest first)
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}

	return chain, nil
}

// ListResponsesByConversation lists a conversation's responses via the
// secondary index, at a cost proportional to the requested page rather than
// the keyspace or even the conversation's full history (see
// listIndexedResponseIDs).
//
// Read path: index exists → read the requested page from it. Otherwise
// check the empty-conversation marker → nothing to do. Otherwise this may
// be a pre-index conversation, an unknown ID, or a genuinely empty one that
// has never been checked before: ensureConversationIndex resolves which,
// running the one-time O(N) legacy scan at most once per conversation. See
// blueprint §5 Phase 3 for the full state diagram this implements.
//
// Order/After/Before parity note: this implementation honors ListOptions.Order
// (default "desc", newest first) and After/Before cursors, per the contract
// documented on ListOptions in interface.go. MemoryStore does not — it
// always returns insertion order regardless of these fields (see its own
// doc comment). Bringing MemoryStore into line is out of scope for #2814;
// callers that need a specific order from either backend today should not
// assume Redis and MemoryStore agree on default order.
func (s *RedisStore) ListResponsesByConversation(ctx context.Context, conversationID string, opts ListOptions) ([]*responseapi.StoredResponse, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if conversationID == "" {
		return nil, ErrInvalidInput
	}

	indexKey := s.conversationIndexKey(conversationID)

	indexExists, err := s.client.Exists(ctx, indexKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to check conversation index: %w", err)
	}
	if indexExists > 0 {
		return s.listIndexedResponses(ctx, conversationID, opts)
	}

	emptyExists, err := s.client.Exists(ctx, s.emptyConversationIndexMarkerKey(conversationID)).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to check conversation index empty marker: %w", err)
	}
	if emptyExists > 0 {
		return nil, nil
	}

	// Neither exists: this conversation has never been resolved. Scan once
	// (behind a migration lock, so concurrent readers of the same missing
	// index don't all scan at once) and recheck.
	if err := s.ensureConversationIndex(ctx, conversationID); err != nil {
		return nil, err
	}

	indexExists, err = s.client.Exists(ctx, indexKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to check conversation index: %w", err)
	}
	if indexExists > 0 {
		return s.listIndexedResponses(ctx, conversationID, opts)
	}

	// The backfill (this call's own, or a concurrent one) found nothing and
	// left the empty marker instead — or, if even that failed, the next call
	// simply scans again. Either way there is nothing to return right now.
	return nil, nil
}

// listIndexedResponses reads one page of a conversation's responses through
// its already-confirmed-existing index: a bounded rank-window read
// (listIndexedResponseIDs), not a full-index scan, so cost is proportional
// to the page requested rather than the conversation's full history.
//
// Because only one page of IDs is read, pruning a stale or moved entry can
// make this return fewer than the requested Limit even when more matching
// responses exist further in the index. That is an accepted Phase 4
// trade-off (blueprint §5 Phase 4): topping up short pages by re-reading
// further windows would turn a bug fix into a pagination redesign.
func (s *RedisStore) listIndexedResponses(ctx context.Context, conversationID string, opts ListOptions) ([]*responseapi.StoredResponse, error) {
	responseIDs, err := s.listIndexedResponseIDs(ctx, conversationID, opts)
	if err != nil {
		return nil, err
	}
	if len(responseIDs) == 0 {
		return nil, nil
	}

	fetched, missingIDs, err := s.fetchResponsesPipelined(ctx, responseIDs)
	if err != nil {
		return nil, err
	}

	// Payloads expire on their own TTL; their index entries do not. Pruning keeps
	// a long-lived conversation's index from growing without bound. Best-effort:
	// a failure here just costs the same prune again on the next listing.
	if err := s.unindexResponse(ctx, conversationID, missingIDs...); err != nil {
		logging.Warnf("RedisStore: failed to prune %d stale index entr(y/ies) from conversation %s: %v",
			len(missingIDs), conversationID, err)
	}

	// Guard against an entry left behind by a response that moved
	// conversation: prune it from this conversation's index too, not just
	// filter it from this page, since it will never legitimately belong here.
	responses := make([]*responseapi.StoredResponse, 0, len(fetched))
	for _, response := range fetched {
		if response.ConversationID == conversationID {
			responses = append(responses, response)
			continue
		}
		if err := s.unindexResponse(ctx, conversationID, response.ID); err != nil {
			logging.Warnf("RedisStore: failed to prune response %s moved out of conversation %s: %v",
				response.ID, conversationID, err)
		}
	}

	return responses, nil
}

// normalizedListOptions is ListOptions after validation and defaulting:
// Limit is always in [1, MaxListLimit], and Order is always exactly "asc"
// or "desc".
type normalizedListOptions struct {
	Limit  int
	Order  string
	After  string
	Before string
}

// normalizeResponseListOptions validates and defaults a caller's ListOptions
// for indexed reads. Order defaults to "desc" (newest first), matching the
// documented ListOptions.Order contract in interface.go and OpenAI's list
// default — a contract neither store implementation actually honored before
// this issue (both simply returned index/insertion order regardless of
// Order). Rejects an unrecognized Order and rejects After and Before set
// together, rather than silently picking one and ignoring the other.
func normalizeResponseListOptions(opts ListOptions) (normalizedListOptions, error) {
	if opts.After != "" && opts.Before != "" {
		return normalizedListOptions{}, ErrInvalidInput
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}
	if limit > MaxListLimit {
		limit = MaxListLimit
	}

	order := opts.Order
	switch order {
	case "":
		order = "desc"
	case "asc", "desc":
		// already valid
	default:
		return normalizedListOptions{}, ErrInvalidInput
	}

	return normalizedListOptions{Limit: limit, Order: order, After: opts.After, Before: opts.Before}, nil
}

// listIndexedResponseIDs reads one bounded window of response IDs from a
// conversation's index: at most normalizeResponseListOptions(opts).Limit
// IDs, in the requested order, optionally positioned after/before a cursor
// response ID — never a full ZRANGE 0 -1.
//
// Cursors are resolved via ZRANK (ascending order) or ZREVRANK (descending
// order), i.e. rank in the order actually being read, and the window is
// then read with the matching ZRANGE/ZREVRANGE. A cursor naming a response
// ID that is not currently a member of the index (evicted, wrong
// conversation, typo'd by the caller) yields an empty page rather than an
// error: the same behavior as an ordinary page with nothing left to return.
func (s *RedisStore) listIndexedResponseIDs(ctx context.Context, conversationID string, opts ListOptions) ([]string, error) {
	normalized, err := normalizeResponseListOptions(opts)
	if err != nil {
		return nil, err
	}

	indexKey := s.conversationIndexKey(conversationID)
	ascending := normalized.Order == "asc"

	rank := func(member string) (int64, bool, error) {
		var cmd *redis.IntCmd
		if ascending {
			cmd = s.client.ZRank(ctx, indexKey, member)
		} else {
			cmd = s.client.ZRevRank(ctx, indexKey, member)
		}
		r, err := cmd.Result()
		if err != nil {
			if errors.Is(err, redis.Nil) {
				return 0, false, nil
			}
			return 0, false, fmt.Errorf("failed to rank conversation index cursor %s: %w", member, err)
		}
		return r, true, nil
	}

	limit := int64(normalized.Limit)
	var start, end int64

	switch {
	case normalized.After != "":
		r, found, err := rank(normalized.After)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, nil
		}
		start = r + 1
		end = start + limit - 1
	case normalized.Before != "":
		r, found, err := rank(normalized.Before)
		if err != nil {
			return nil, err
		}
		if !found {
			return nil, nil
		}
		end = r - 1
		start = end - limit + 1
		if start < 0 {
			start = 0
		}
	default:
		start = 0
		end = limit - 1
	}

	if end < start {
		return nil, nil
	}

	var idsCmd *redis.StringSliceCmd
	if ascending {
		idsCmd = s.client.ZRange(ctx, indexKey, start, end)
	} else {
		idsCmd = s.client.ZRevRange(ctx, indexKey, start, end)
	}
	ids, err := idsCmd.Result()
	if err != nil {
		return nil, fmt.Errorf("failed to read conversation index window: %w", err)
	}

	return ids, nil
}

// Conversation Store Methods

func (s *RedisStore) CreateConversation(ctx context.Context, conversation *responseapi.StoredConversation) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if conversation == nil || conversation.ID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ConversationKeyPrefix + conversation.ID)

	exists, err := s.client.Exists(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to check conversation existence: %w", err)
	}
	if exists > 0 {
		return ErrAlreadyExists
	}

	data, err := json.Marshal(conversation)
	if err != nil {
		return fmt.Errorf("failed to serialize conversation: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to store conversation in Redis: %w", err)
	}

	return nil
}

func (s *RedisStore) GetConversation(ctx context.Context, conversationID string) (*responseapi.StoredConversation, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if conversationID == "" {
		return nil, ErrInvalidInput
	}

	key := s.buildKey(ConversationKeyPrefix + conversationID)

	data, err := s.client.Get(ctx, key).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to get conversation from Redis: %w", err)
	}

	var conversation responseapi.StoredConversation
	if err := json.Unmarshal(data, &conversation); err != nil {
		return nil, fmt.Errorf("failed to deserialize conversation: %w", err)
	}

	return &conversation, nil
}

func (s *RedisStore) UpdateConversation(ctx context.Context, conversation *responseapi.StoredConversation) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if conversation == nil || conversation.ID == "" {
		return ErrInvalidInput
	}

	key := s.buildKey(ConversationKeyPrefix + conversation.ID)

	exists, err := s.client.Exists(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to check conversation existence: %w", err)
	}
	if exists == 0 {
		return ErrNotFound
	}

	data, err := json.Marshal(conversation)
	if err != nil {
		return fmt.Errorf("failed to serialize conversation: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to update conversation in Redis: %w", err)
	}

	return nil
}

func (s *RedisStore) DeleteConversation(ctx context.Context, conversationID string, deleteResponses bool) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if conversationID == "" {
		return ErrInvalidInput
	}

	convKey := s.buildKey(ConversationKeyPrefix + conversationID)
	deleted, err := s.client.Del(ctx, convKey).Result()
	if err != nil {
		return fmt.Errorf("failed to delete conversation from Redis: %w", err)
	}
	if deleted == 0 {
		return ErrNotFound
	}

	// Optionally delete all responses in the conversation
	if deleteResponses {
		if err := s.deleteConversationResponses(ctx, conversationID); err != nil {
			return err
		}
	}

	return nil
}

// deleteConversationResponses removes a conversation's responses and its index.
// Reads the index directly: going via ListResponsesByConversation would cap the
// cascade at the pagination limit.
func (s *RedisStore) deleteConversationResponses(ctx context.Context, conversationID string) error {
	indexKey := s.conversationIndexKey(conversationID)

	responseIDs, err := s.client.ZRange(ctx, indexKey, 0, -1).Result()
	if err != nil {
		return fmt.Errorf("failed to list responses for deletion: %w", err)
	}

	pipe := s.client.Pipeline()
	for _, responseID := range responseIDs {
		pipe.Del(ctx, s.buildKey(ResponseKeyPrefix+responseID))
	}
	pipe.Del(ctx, indexKey)

	if _, err := pipe.Exec(ctx); err != nil {
		// Conversation is already gone; a returned error could not be usefully retried.
		logging.Warnf("RedisStore: failed to delete %d response(s) of conversation %s: %v",
			len(responseIDs), conversationID, err)
	}

	return nil
}

func (s *RedisStore) ListConversations(ctx context.Context, opts ListOptions) ([]*responseapi.StoredConversation, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}

	pattern := s.buildKey(ConversationKeyPrefix + "*")
	var conversations []*responseapi.StoredConversation

	iter := s.client.Scan(ctx, 0, pattern, 0).Iterator()
	for iter.Next(ctx) {
		key := iter.Val()

		data, err := s.client.Get(ctx, key).Bytes()
		if err != nil {
			continue
		}

		var conversation responseapi.StoredConversation
		if err := json.Unmarshal(data, &conversation); err != nil {
			continue
		}

		conversations = append(conversations, &conversation)
	}

	if err := iter.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan conversations: %w", err)
	}

	// Apply list options (limit, pagination)
	conversations = ApplyConvListOptions(conversations, opts)

	return conversations, nil
}

// AddResponseToConversation adds a response ID to a conversation.
func (s *RedisStore) AddResponseToConversation(ctx context.Context, conversationID, responseID string) error {
	if !s.enabled {
		return ErrStoreDisabled
	}
	if conversationID == "" || responseID == "" {
		return ErrInvalidInput
	}

	// Membership is recorded by StoreResponse via the conversation index, and
	// traversal by previous_response_id. Kept for conversation metadata updates.
	return nil
}

// Helper methods

// indexResponse adds a response to its conversation index, scored by
// created_at; refreshes the index TTL; and clears the empty-conversation
// marker so a newly indexed write is never hidden behind a stale "scanned,
// found nothing" marker (checked here best-effort — a marker left behind by
// this call failing is harmless, because every read checks the index before
// the marker).
//
// Returns an error instead of swallowing it: the payload this indexes is
// already durable by the time this runs (StoreResponse writes it first), so
// the caller — not this helper — must decide what an index failure means:
// StoreResponse rolls the payload back, UpdateResponse restores the previous
// payload, DeleteResponse and lazy backfill may choose to log and continue.
func (s *RedisStore) indexResponse(ctx context.Context, conversationID, responseID string, createdAt int64) error {
	if conversationID == "" || responseID == "" {
		return nil
	}
	if s.indexResponseOverride != nil {
		return s.indexResponseOverride(ctx, conversationID, responseID, createdAt)
	}

	indexKey := s.conversationIndexKey(conversationID)

	pipe := s.client.Pipeline()
	pipe.ZAdd(ctx, indexKey, redis.Z{Score: float64(createdAt), Member: responseID})
	if s.ttl > 0 {
		// Outlive the newest member. Guarded: EXPIRE with 0 deletes the key.
		pipe.Expire(ctx, indexKey, s.ttl)
	}
	// Single-key DEL, never combined with another key: Cluster safe. A no-op
	// (returns 0, not an error) when no marker exists.
	pipe.Del(ctx, s.emptyConversationIndexMarkerKey(conversationID))

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to index response %s in conversation %s: %w", responseID, conversationID, err)
	}

	return nil
}

// unindexResponse drops response IDs from a conversation index. ZREM is
// variadic but touches only one key (the zset), all members belong to the
// same conversation index, so it stays Cluster safe.
func (s *RedisStore) unindexResponse(ctx context.Context, conversationID string, responseIDs ...string) error {
	if conversationID == "" || len(responseIDs) == 0 {
		return nil
	}

	members := make([]interface{}, len(responseIDs))
	for i, responseID := range responseIDs {
		members[i] = responseID
	}

	if err := s.client.ZRem(ctx, s.conversationIndexKey(conversationID), members...).Err(); err != nil {
		return fmt.Errorf("failed to remove %d response(s) from conversation %s index: %w", len(responseIDs), conversationID, err)
	}

	return nil
}

// legacyBackfillResult reports how many responses a lazy backfill scan found
// and indexed for a conversation, mainly for logging.
type legacyBackfillResult struct {
	// Found is the number of responses discovered belonging to the scanned
	// conversation. Zero means the empty marker was set instead.
	Found int
}

// indexOrMarkerExists reports whether a conversation's index or its empty
// marker exists, via two single-key EXISTS calls. Deliberately not a
// variadic Exists(ctx, indexKey, markerKey): the two keys have different
// prefixes and are not guaranteed to share a Cluster hash slot.
func (s *RedisStore) indexOrMarkerExists(ctx context.Context, conversationID string) (bool, error) {
	indexExists, err := s.client.Exists(ctx, s.conversationIndexKey(conversationID)).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check conversation index: %w", err)
	}
	if indexExists > 0 {
		return true, nil
	}

	emptyExists, err := s.client.Exists(ctx, s.emptyConversationIndexMarkerKey(conversationID)).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check conversation index empty marker: %w", err)
	}

	return emptyExists > 0, nil
}

// ensureConversationIndex guarantees that, barring a concurrent delete of
// both immediately afterward, either the conversation's index or its empty
// marker exists once this returns without error. It runs the O(N) legacy
// scan (lazyBackfillConversationIndex) at most once per conversation per
// marker/index lifetime.
//
// A short migration lock (SET NX PX conversationIndexMigrationLockTTL)
// keeps concurrent readers of the same missing index from all scanning at
// once, but holding it is an optimization, not a correctness dependency: a
// reader that cannot acquire it backs off briefly, rechecks, and — if the
// holder appears to have died mid-scan (pod restart, deadline) rather than
// finished — runs the scan itself anyway. Every path converges on the same
// additive, idempotent backfill.
func (s *RedisStore) ensureConversationIndex(ctx context.Context, conversationID string) error {
	lockKey := s.conversationIndexLockKey(conversationID)
	token := []byte(fmt.Sprintf("%d:%d", time.Now().UnixNano(), os.Getpid()))

	acquired, err := s.client.SetNX(ctx, lockKey, token, conversationIndexMigrationLockTTL).Result()
	if err != nil {
		return fmt.Errorf("failed to acquire conversation index migration lock: %w", err)
	}

	if acquired {
		defer func() {
			// Reused single-key compare-delete: never releases a lock this
			// call didn't acquire (e.g. one that already expired and was
			// re-acquired by someone else).
			if _, releaseErr := s.compareDeleteResponsePayload(ctx, lockKey, token); releaseErr != nil {
				logging.Debugf("RedisStore: failed to release conversation index migration lock for %s: %v",
					conversationID, releaseErr)
			}
		}()

		// Recheck under the lock: a writer, or a backfill that started just
		// before this one acquired it, may have already resolved this
		// conversation.
		exists, existsErr := s.indexOrMarkerExists(ctx, conversationID)
		if existsErr != nil {
			return existsErr
		}
		if exists {
			return nil
		}

		result, backfillErr := s.lazyBackfillConversationIndex(ctx, conversationID)
		if backfillErr != nil {
			return backfillErr
		}
		logging.Debugf("RedisStore: lazy-backfilled conversation %s index with %d response(s)",
			conversationID, result.Found)
		return nil
	}

	// Someone else holds the lock. Back off briefly and recheck; if the
	// index/marker still hasn't appeared once the bounded wait is over, the
	// holder may have died mid-scan — run the scan locally rather than block
	// indefinitely. Correctness wins over avoiding duplicate work, which is
	// the lock's only job.
	const (
		lockWaitAttempts = 5
		lockWaitDelay    = 100 * time.Millisecond
	)
	for i := 0; i < lockWaitAttempts; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(lockWaitDelay):
		}

		exists, existsErr := s.indexOrMarkerExists(ctx, conversationID)
		if existsErr != nil {
			return existsErr
		}
		if exists {
			return nil
		}
	}

	result, err := s.lazyBackfillConversationIndex(ctx, conversationID)
	if err != nil {
		return err
	}
	logging.Debugf("RedisStore: lazy-backfilled conversation %s index with %d response(s) after lock wait",
		conversationID, result.Found)
	return nil
}

// lazyBackfillConversationIndex performs the one-time O(N) scan that makes a
// pre-index conversation's responses discoverable: it walks every response
// payload once, keeps the ones matching conversationID, and either indexes
// them or — if none match — sets the empty marker so the next read does not
// scan again.
//
// Idempotent and additive: it only ZADDs discovered members and never
// deletes, so a concurrently indexed write (e.g. a StoreResponse racing this
// scan) is never undone by it, no matter which finishes first.
func (s *RedisStore) lazyBackfillConversationIndex(ctx context.Context, conversationID string) (legacyBackfillResult, error) {
	if s.lazyBackfillPreScanHook != nil {
		s.lazyBackfillPreScanHook()
	}

	type discovered struct {
		id        string
		createdAt int64
	}
	var found []discovered

	err := s.scanResponsePayloads(ctx, func(batch []*responseapi.StoredResponse) error {
		for _, response := range batch {
			if response.ConversationID == conversationID {
				found = append(found, discovered{id: response.ID, createdAt: response.CreatedAt})
			}
		}
		return nil
	})
	if err != nil {
		return legacyBackfillResult{}, fmt.Errorf("failed to backfill conversation index: %w", err)
	}

	if len(found) == 0 {
		if err := s.client.Set(ctx, s.emptyConversationIndexMarkerKey(conversationID), "v1", s.ttl).Err(); err != nil {
			// Correctness is preserved either way: the next read scans
			// again rather than trusting a marker that isn't there.
			logging.Debugf("RedisStore: failed to set empty conversation index marker for %s: %v",
				conversationID, err)
		}
		return legacyBackfillResult{Found: 0}, nil
	}

	// Deterministic order: primary by created_at, tie-broken by response ID
	// (blueprint §3.6) — never fabricate sub-second ordering from wall clock
	// time during repair.
	sort.Slice(found, func(i, j int) bool {
		if found[i].createdAt != found[j].createdAt {
			return found[i].createdAt < found[j].createdAt
		}
		return found[i].id < found[j].id
	})

	indexKey := s.conversationIndexKey(conversationID)
	for start := 0; start < len(found); start += redisBackfillBatchSize {
		end := start + redisBackfillBatchSize
		if end > len(found) {
			end = len(found)
		}

		members := make([]redis.Z, end-start)
		for i, d := range found[start:end] {
			members[i] = redis.Z{Score: float64(d.createdAt), Member: d.id}
		}
		if err := s.client.ZAdd(ctx, indexKey, members...).Err(); err != nil {
			return legacyBackfillResult{}, fmt.Errorf("failed to backfill conversation index: %w", err)
		}
	}

	if s.ttl > 0 {
		if err := s.client.Expire(ctx, indexKey, s.ttl).Err(); err != nil {
			logging.Warnf("RedisStore: failed to refresh TTL on backfilled conversation index %s: %v",
				conversationID, err)
		}
	}

	// Best-effort: the index now exists, and every read checks it before the
	// marker, so a marker left behind here is harmless even if this DEL fails.
	if err := s.client.Del(ctx, s.emptyConversationIndexMarkerKey(conversationID)).Err(); err != nil {
		logging.Debugf("RedisStore: failed to clear empty conversation index marker for %s: %v",
			conversationID, err)
	}

	return legacyBackfillResult{Found: len(found)}, nil
}

// scanResponsePayloads walks every response payload key exactly once,
// decoding each into a StoredResponse and delivering them to visit in
// bounded batches (redisBackfillBatchSize).
//
// Cluster-aware: a single Redis Cluster node's keyspace only holds the slots
// assigned to it, so in Cluster mode this scans every master via
// ForEachMaster. Standalone mode scans the one client directly.
//
// Used only by lazy legacy backfill — this is the O(N) operation the index
// exists to avoid on the hot read path.
func (s *RedisStore) scanResponsePayloads(ctx context.Context, visit func(batch []*responseapi.StoredResponse) error) error {
	s.scanInvocations++

	pattern := s.buildKey(ResponseKeyPrefix + "*")

	scanNode := func(ctx context.Context, client redis.UniversalClient) error {
		var keys []string
		flush := func() error {
			if len(keys) == 0 {
				return nil
			}
			batch := s.getResponsesPipelined(ctx, client, keys)
			keys = keys[:0]
			if len(batch) == 0 {
				return nil
			}
			return visit(batch)
		}

		iter := client.Scan(ctx, 0, pattern, redisScanCount).Iterator()
		for iter.Next(ctx) {
			keys = append(keys, iter.Val())
			if len(keys) >= redisBackfillBatchSize {
				if err := flush(); err != nil {
					return err
				}
			}
		}
		if err := iter.Err(); err != nil {
			return fmt.Errorf("failed to scan response keys: %w", err)
		}

		return flush()
	}

	if clusterClient, ok := s.client.(*redis.ClusterClient); ok {
		return clusterClient.ForEachMaster(ctx, func(ctx context.Context, master *redis.Client) error {
			return scanNode(ctx, master)
		})
	}

	return scanNode(ctx, s.client)
}

// getResponsesPipelined GETs and decodes payloads for a batch of already-
// prefixed keys in one round trip against the given client. Malformed or
// missing payloads are skipped with a log line rather than failing the
// batch: a legacy scan must make forward progress even if one record is
// corrupt or expired mid-scan.
func (s *RedisStore) getResponsesPipelined(ctx context.Context, client redis.UniversalClient, keys []string) []*responseapi.StoredResponse {
	if len(keys) == 0 {
		return nil
	}

	pipe := client.Pipeline()
	cmds := make([]*redis.StringCmd, len(keys))
	for i, key := range keys {
		cmds[i] = pipe.Get(ctx, key)
	}
	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		logging.Debugf("RedisStore: scan pipeline execution completed with some errors: %v", err)
	}

	responses := make([]*responseapi.StoredResponse, 0, len(keys))
	for i, cmd := range cmds {
		data, err := cmd.Bytes()
		if err != nil {
			if !errors.Is(err, redis.Nil) {
				logging.Warnf("RedisStore: failed to get response at key %s during scan: %v", keys[i], err)
			}
			continue
		}

		var response responseapi.StoredResponse
		if err := json.Unmarshal(data, &response); err != nil {
			logging.Warnf("RedisStore: failed to parse response at key %s during scan: %v", keys[i], err)
			continue
		}
		if response.ID == "" {
			continue
		}

		responses = append(responses, &response)
	}

	return responses
}

// storedConversationID reports a response's current conversation so update and
// delete can repair the index, and returns ErrNotFound when it is absent. An
// unreadable payload is not fatal, it only costs the old index entry.
func (s *RedisStore) storedConversationID(ctx context.Context, responseID string) (string, error) {
	data, err := s.client.Get(ctx, s.buildKey(ResponseKeyPrefix+responseID)).Bytes()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return "", ErrNotFound
		}
		return "", fmt.Errorf("failed to check response existence: %w", err)
	}

	var stored responseapi.StoredResponse
	if err := json.Unmarshal(data, &stored); err != nil {
		logging.Warnf("RedisStore: failed to parse stored response %s while updating its conversation index: %v",
			responseID, err)
		return "", nil
	}

	return stored.ConversationID, nil
}

func (s *RedisStore) collectChainIDs(ctx context.Context, startID string) ([]string, error) {
	var responseIDs []string
	currentID := startID
	visited := make(map[string]bool)

	// Maximum chain length to prevent infinite loops
	const maxChainLength = 1000

	for currentID != "" && len(responseIDs) < maxChainLength {
		// Prevent circular references
		if visited[currentID] {
			logging.Warnf("RedisStore: circular reference detected at %s", currentID)
			break
		}
		visited[currentID] = true

		responseIDs = append(responseIDs, currentID)

		response, err := s.GetResponse(ctx, currentID)
		if err != nil {
			if errors.Is(err, ErrNotFound) {
				// If this is the first response (start of chain), return error
				if len(responseIDs) == 1 {
					return nil, ErrNotFound
				}
				// Otherwise, just break - the chain ended early
				logging.Warnf("RedisStore: response %s not found in chain", currentID)
				break
			}
			return nil, fmt.Errorf("failed to fetch response %s: %w", currentID, err)
		}

		currentID = response.PreviousResponseID
	}

	return responseIDs, nil
}

// fetchResponsesPipelined loads response IDs in one round trip, also returning
// the IDs whose payload is gone so index-driven callers can prune them.
func (s *RedisStore) fetchResponsesPipelined(ctx context.Context, responseIDs []string) ([]*responseapi.StoredResponse, []string, error) {
	if len(responseIDs) == 0 {
		return []*responseapi.StoredResponse{}, nil, nil
	}

	pipe := s.client.Pipeline()

	cmds := make([]*redis.StringCmd, len(responseIDs))
	for i, id := range responseIDs {
		key := s.buildKey(ResponseKeyPrefix + id)
		cmds[i] = pipe.Get(ctx, key)
	}

	_, err := pipe.Exec(ctx)
	if err != nil && !errors.Is(err, redis.Nil) {
		// Some commands might fail, but we continue to process successful ones
		logging.Debugf("RedisStore: pipeline execution completed with some errors: %v", err)
	}

	// Process results
	var (
		found      []*responseapi.StoredResponse
		missingIDs []string
	)
	for i, cmd := range cmds {
		data, err := cmd.Bytes()
		if err != nil {
			if errors.Is(err, redis.Nil) {
				logging.Warnf("RedisStore: response %s not found (may have expired)", responseIDs[i])
				missingIDs = append(missingIDs, responseIDs[i])
				continue
			}
			logging.Warnf("RedisStore: failed to get response %s: %v", responseIDs[i], err)
			continue
		}

		var response responseapi.StoredResponse
		if err := json.Unmarshal(data, &response); err != nil {
			logging.Warnf("RedisStore: failed to parse response %s: %v", responseIDs[i], err)
			continue
		}

		found = append(found, &response)
	}

	return found, missingIDs, nil
}
