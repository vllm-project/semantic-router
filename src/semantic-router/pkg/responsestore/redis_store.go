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
)

// The function validates configuration, establishes connection, and tests connectivity.
func NewRedisStore(config StoreConfig) (*RedisStore, error) {
	logging.Debugf("RedisStore: Initializing with cluster_mode=%v, config_path=%s",
		config.Redis.ClusterMode, config.Redis.ConfigPath)

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

	logging.Infof("RedisStore: initialized successfully (cluster_mode=%v, key_prefix=%s, ttl=%s)",
		finalCfg.ClusterMode, keyPrefix, ttl)

	return store, nil
}

func loadRedisStoreConfig(cfg RedisStoreConfig) (RedisStoreConfig, error) {
	// If no external config, return inline config as-is
	if cfg.ConfigPath == "" {
		logging.Debugf("RedisStore: using inline configuration")
		return cfg, nil
	}

	// Load external configuration
	logging.Debugf("RedisStore: loading config from file: %s", cfg.ConfigPath)

	data, err := os.ReadFile(cfg.ConfigPath)
	if err != nil {
		return cfg, fmt.Errorf("failed to read config file %s: %w", cfg.ConfigPath, err)
	}

	var fileCfg RedisStoreConfig
	if err := yaml.Unmarshal(data, &fileCfg); err != nil {
		return cfg, fmt.Errorf("failed to parse config file %s: %w", cfg.ConfigPath, err)
	}

	logging.Debugf("RedisStore: external config loaded (address=%s, cluster_mode=%v)",
		fileCfg.Address, fileCfg.ClusterMode)

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

		logging.Debugf("RedisStore: TLS enabled")
	}

	// Create client based on mode
	if cfg.ClusterMode {
		logging.Infof("RedisStore: creating cluster client (nodes=%d, pool_size=%d)",
			len(cfg.ClusterAddresses), cfg.PoolSize)

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
	logging.Infof("RedisStore: creating standalone client (address=%s, db=%d, pool_size=%d)",
		cfg.Address, cfg.DB, cfg.PoolSize)

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

	exists, err := s.client.Exists(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to check response existence: %w", err)
	}
	if exists > 0 {
		return ErrAlreadyExists
	}

	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to store response in Redis: %w", err)
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

	exists, err := s.client.Exists(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to check response existence: %w", err)
	}
	if exists == 0 {
		return ErrNotFound
	}

	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	if err := s.client.Set(ctx, key, data, s.ttl).Err(); err != nil {
		return fmt.Errorf("failed to update response in Redis: %w", err)
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

	deleted, err := s.client.Del(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to delete response from Redis: %w", err)
	}
	if deleted == 0 {
		return ErrNotFound
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
	chain, err := s.fetchResponsesPipelined(ctx, responseIDs)
	if err != nil {
		return nil, err
	}

	// Phase 3: Reverse chain to get chronological order (oldest first)
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}

	return chain, nil
}

func (s *RedisStore) ListResponsesByConversation(ctx context.Context, conversationID string, opts ListOptions) ([]*responseapi.StoredResponse, error) {
	if !s.enabled {
		return nil, ErrStoreDisabled
	}
	if conversationID == "" {
		return nil, ErrInvalidInput
	}

	// Use SCAN to find all response keys
	pattern := s.buildKey(ResponseKeyPrefix + "*")
	var responses []*responseapi.StoredResponse

	iter := s.client.Scan(ctx, 0, pattern, 0).Iterator()
	for iter.Next(ctx) {
		key := iter.Val()

		// Get response
		data, err := s.client.Get(ctx, key).Bytes()
		if err != nil {
			continue // Skip errors (key might have expired)
		}

		var response responseapi.StoredResponse
		if err := json.Unmarshal(data, &response); err != nil {
			continue
		}

		if response.ConversationID == conversationID {
			responses = append(responses, &response)
		}
	}

	if err := iter.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan responses: %w", err)
	}

	responses = ApplyListOptions(responses, opts)

	return responses, nil
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
		responses, err := s.ListResponsesByConversation(ctx, conversationID, ListOptions{})
		if err != nil {
			return fmt.Errorf("failed to list responses for deletion: %w", err)
		}

		for _, resp := range responses {
			if err := s.DeleteResponse(ctx, resp.ID); err != nil && !errors.Is(err, ErrNotFound) {
				logging.Warnf("RedisStore: failed to delete response %s: %v", resp.ID, err)
			}
		}
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

	// This is automatically handled by the conversation chain via previous_response_id
	// This method can be used to update conversation metadata if needed
	return nil
}

// Helper methods

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

func (s *RedisStore) fetchResponsesPipelined(ctx context.Context, responseIDs []string) ([]*responseapi.StoredResponse, error) {
	if len(responseIDs) == 0 {
		return []*responseapi.StoredResponse{}, nil
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
	var chain []*responseapi.StoredResponse
	for i, cmd := range cmds {
		data, err := cmd.Bytes()
		if err != nil {
			if errors.Is(err, redis.Nil) {
				logging.Warnf("RedisStore: response %s not found (may have expired)", responseIDs[i])
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

		chain = append(chain, &response)
	}

	return chain, nil
}
