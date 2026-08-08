package responsestore

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"os"
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
}

const (
	// ResponseKeyPrefix for response keys
	// Combined with key_prefix (default "sr:"): sr:response:resp_xxxxx
	ResponseKeyPrefix = "response:"

	// ConversationKeyPrefix for conversation keys
	// Combined with key_prefix (default "sr:"): sr:conversation:conv_xxxxx
	ConversationKeyPrefix = "conversation:"

	// ConversationIndexKeyPrefix for the secondary index that maps a conversation
	// to the responses it contains. The index is a sorted set scored by the
	// response created_at, so listing a conversation costs one ZRANGE plus one
	// pipelined GET per member instead of a scan over the whole keyspace.
	// Combined with key_prefix (default "sr:"): sr:conversation-index:conv_xxxxx
	//
	// The prefix intentionally does not start with ConversationKeyPrefix so index
	// keys are not matched by the sr:conversation:* scan in ListConversations.
	ConversationIndexKeyPrefix = "conversation-index:"
)

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

// conversationIndexKey returns the key of the sorted set indexing the responses
// that belong to a conversation.
func (s *RedisStore) conversationIndexKey(conversationID string) string {
	return s.buildKey(ConversationIndexKeyPrefix + conversationID)
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

	// SET NX enforces the "must not already exist" contract in a single atomic
	// command, and guarantees the payload is readable before the ID is published
	// to the conversation index below.
	stored, err := s.client.SetNX(ctx, key, data, s.ttl).Result()
	if err != nil {
		return fmt.Errorf("failed to store response in Redis: %w", err)
	}
	if !stored {
		return ErrAlreadyExists
	}

	s.indexResponse(ctx, response.ConversationID, response.ID, response.CreatedAt)

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

	// Reading the stored conversation doubles as the existence check and tells us
	// whether the response is moving between conversations.
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

	if previousConversationID != "" && previousConversationID != response.ConversationID {
		s.unindexResponse(ctx, previousConversationID, response.ID)
	}
	s.indexResponse(ctx, response.ConversationID, response.ID, response.CreatedAt)

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

	// Read the conversation first so its index entry can be dropped along with
	// the payload; a missing key already means there is nothing to delete.
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

	s.unindexResponse(ctx, conversationID, responseID)

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

// ListResponsesByConversation lists a conversation's responses through the
// secondary index maintained by the write paths.
//
// Only responses written by a router that maintains that index are listed here.
// Responses persisted before the index existed stay fully retrievable by ID and
// through GetConversationChain, and age out on their own TTL (30 days by default).
func (s *RedisStore) ListResponsesByConversation(ctx context.Context, conversationID string, opts ListOptions) ([]*responseapi.StoredResponse, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if conversationID == "" {
		return nil, ErrInvalidInput
	}

	// The secondary index yields exactly this conversation's response IDs, in
	// chronological order, so the work is proportional to the conversation rather
	// than to the whole keyspace.
	responseIDs, err := s.client.ZRange(ctx, s.conversationIndexKey(conversationID), 0, -1).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to read conversation index: %w", err)
	}
	if len(responseIDs) == 0 {
		return nil, nil
	}

	fetched, missingIDs, err := s.fetchResponsesPipelined(ctx, responseIDs)
	if err != nil {
		return nil, err
	}

	// Payloads expire on their own TTL while index entries do not, so prune the
	// entries we just found to be dead. The index cannot outlive its newest
	// member, but pruning keeps a long-lived conversation's index bounded.
	s.unindexResponse(ctx, conversationID, missingIDs...)

	// Guard against an entry left behind by a response that moved conversation.
	var responses []*responseapi.StoredResponse
	for _, response := range fetched {
		if response.ConversationID == conversationID {
			responses = append(responses, response)
		}
	}

	return ApplyListOptions(responses, opts), nil
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

// deleteConversationResponses removes every response indexed for a conversation
// along with the index itself. It reads the index directly rather than going
// through ListResponsesByConversation so the pagination limit cannot leave
// responses behind.
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
		// The conversation itself is already gone, so report the leftovers instead
		// of failing a delete the caller cannot usefully retry.
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

	// Membership is recorded by StoreResponse, which indexes the response under
	// its ConversationID, and traversal is handled by previous_response_id.
	// This method can be used to update conversation metadata if needed.
	return nil
}

// Helper methods

// indexResponse publishes a response ID into its conversation's secondary index,
// scored by creation time so the index reads back in chronological order.
//
// Failures are logged rather than returned: the payload is already stored and
// stays reachable by ID and through its chain, so failing the whole write (which
// the caller could only retry into ErrAlreadyExists) would be worse than leaving
// this one response out of the conversation listing.
func (s *RedisStore) indexResponse(ctx context.Context, conversationID, responseID string, createdAt int64) {
	if conversationID == "" {
		return
	}

	indexKey := s.conversationIndexKey(conversationID)

	pipe := s.client.Pipeline()
	pipe.ZAdd(ctx, indexKey, redis.Z{Score: float64(createdAt), Member: responseID})
	if s.ttl > 0 {
		// Keep the index alive at least as long as its newest member. Guarded
		// because EXPIRE with a zero TTL deletes the key outright.
		pipe.Expire(ctx, indexKey, s.ttl)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		logging.Warnf("RedisStore: failed to index response %s in conversation %s: %v",
			responseID, conversationID, err)
	}
}

// unindexResponse drops response IDs from a conversation's secondary index.
// Like indexResponse it is best-effort: a stale entry only costs one extra GET
// on the next listing, which then prunes it.
func (s *RedisStore) unindexResponse(ctx context.Context, conversationID string, responseIDs ...string) {
	if conversationID == "" || len(responseIDs) == 0 {
		return
	}

	members := make([]interface{}, len(responseIDs))
	for i, responseID := range responseIDs {
		members[i] = responseID
	}

	if err := s.client.ZRem(ctx, s.conversationIndexKey(conversationID), members...).Err(); err != nil {
		logging.Warnf("RedisStore: failed to remove %d response(s) from conversation %s index: %v",
			len(responseIDs), conversationID, err)
	}
}

// storedConversationID reports which conversation a response currently belongs
// to, so update and delete can repair the index they invalidate. It doubles as
// the existence check for those paths and returns ErrNotFound when the response
// is absent. An unreadable payload is not fatal: the caller may still overwrite
// or delete the key, it just cannot repair the old index entry.
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

// fetchResponsesPipelined loads the given response IDs in a single round trip.
// Alongside the responses it found, it returns the IDs whose payload no longer
// exists, which lets index-driven callers prune their stale entries.
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
