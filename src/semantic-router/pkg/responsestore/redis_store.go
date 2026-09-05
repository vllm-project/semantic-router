package responsestore

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
	"sigs.k8s.io/yaml"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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
	//
	// Atomic: scanResponsePayloads is a production code path reachable
	// concurrently from multiple simultaneous requests missing the same or
	// different conversations' indexes, so a plain int would race under
	// -race (and, more to the point, under real concurrent traffic) even
	// though only tests ever read the value back.
	scanInvocations atomic.Int64

	// lazyBackfillPreScanHook, when non-nil, runs once at the top of
	// lazyBackfillConversationIndex before the scan starts. Unexported
	// test-only seam for deterministically landing a concurrent indexed
	// write in the window the blueprint calls out (§2.2 Redis concurrency):
	// a writer's SETNX + index ZADD completing while a lazy scan for the
	// same conversation is in flight must not be undone by the scan.
	lazyBackfillPreScanHook func()

	// deleteResponseBatchOverride, when non-nil, replaces
	// deleteConversationResponses' per-batch response payload deletion.
	// Unexported test-only seam for asserting that a cascade delete failure
	// partway through is reported to the caller (and leaves the conversation
	// record and remaining batches untouched) without needing a real Redis
	// fault to land mid-pipeline.
	deleteResponseBatchOverride func(ctx context.Context, responseIDs []string) error
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

	// ConversationIndexMigratedKeyPrefix marks a conversation ID for which a
	// legacy-scan backfill has completed — found responses to index, or
	// confirmed none exist: sr:conversation-index-migrated:conv_xxxxx,
	// value "v1".
	//
	// This is deliberately a *different* signal from "the index key exists":
	// a conversation can accumulate real indexed members from ordinary
	// post-upgrade StoreResponse writes long before any backfill scan ever
	// runs for it (e.g. a pre-existing legacy conversation's first
	// post-upgrade turn). Treating index-existence alone as "fully migrated"
	// would let that write's indexResponse call create the index with only
	// the new member, after which every future read would trust that index
	// as complete and never discover the older, still-unindexed responses —
	// silently and permanently. This marker is the only thing
	// ListResponsesByConversation and cascade delete trust to mean "the
	// index (or its absence) is exhaustive as of now"; see
	// ensureConversationIndex and conversationMigrated.
	//
	// Also prevents a caller from forcing repeated full keyspace scans by
	// repeatedly listing the same empty or unknown conversation. Must not
	// start with ConversationKeyPrefix or ResponseKeyPrefix, for the same
	// scan-isolation reason as ConversationIndexKeyPrefix.
	ConversationIndexMigratedKeyPrefix = "conversation-index-migrated:"

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

	// emptyConversationIndexMarkerMaxTTL caps how long a migrated marker
	// survives when its backfill found nothing to index, independent of the
	// store's data-retention TTL (s.ttl, which can be a day or 30 days).
	// Confirming a conversation empty is a weaker, more perishable claim
	// than confirming what its live responses are: an indexing-unaware
	// writer (see ConversationIndexMigratedKeyPrefix) could still land a
	// response into that same conversation later, and this bounds how long
	// such a write can stay hidden behind a stale "confirmed empty" result.
	//
	// Does not apply once the backfill actually finds responses: that
	// marker gets the full store TTL instead (markConversationMigrated),
	// since there is no analogous blind spot once real data has already
	// been discovered and indexed.
	emptyConversationIndexMarkerMaxTTL = 5 * time.Minute

	// redisScanCount is the SCAN COUNT hint used when walking the response
	// keyspace for lazy legacy backfill. A hint, not a hard limit — Redis may
	// return more or fewer keys per cursor step.
	redisScanCount = 1000

	// redisBackfillBatchSize bounds how many keys are GET-pipelined, and how
	// many discovered members are ZADD-ed, per round trip during lazy legacy
	// backfill — keeps a single conversation's backfill from building one
	// unbounded pipeline or command.
	redisBackfillBatchSize = 256

	// redisDeleteBatchSize bounds how many responses are deleted per round
	// trip when cascading a conversation delete, so deleting a very large
	// conversation never needs one ZRANGE 0 -1 or one pipeline sized to the
	// whole conversation.
	redisDeleteBatchSize = 256
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

// conversationIndexMigratedKey returns the marker set once a legacy-scan
// backfill has completed for a conversation, whether or not it found
// anything to index. Its presence — not the index key's — is what makes the
// index's current state (populated or absent) trustworthy as exhaustive;
// see ConversationIndexMigratedKeyPrefix and conversationMigrated.
func (s *RedisStore) conversationIndexMigratedKey(conversationID string) string {
	return s.buildKey(ConversationIndexMigratedKeyPrefix + conversationID)
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
