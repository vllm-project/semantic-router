package responsestore

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

type redisStoreConfigTestCase struct {
	name        string
	config      StoreConfig
	expectError bool
	errorMsg    string
}

func buildRedisStoreConfigTests() []redisStoreConfigTestCase {
	return []redisStoreConfigTestCase{
		{
			name: "valid standalone config",
			config: StoreConfig{
				Enabled:     true,
				TTLSeconds:  3600,
				BackendType: RedisStoreType,
				Redis: RedisStoreConfig{
					Address: "localhost:6379",
					DB:      0,
				},
			},
			expectError: false,
		},
		{
			name: "valid cluster config",
			config: StoreConfig{
				Enabled:     true,
				TTLSeconds:  3600,
				BackendType: RedisStoreType,
				Redis: RedisStoreConfig{
					ClusterMode:      true,
					ClusterAddresses: []string{"node1:6379", "node2:6379"},
					DB:               0,
				},
			},
			expectError: false,
		},
		{
			name: "cluster with non-zero DB",
			config: StoreConfig{
				Enabled:     true,
				TTLSeconds:  3600,
				BackendType: RedisStoreType,
				Redis: RedisStoreConfig{
					ClusterMode:      true,
					ClusterAddresses: []string{"node1:6379"},
					DB:               1,
				},
			},
			expectError: true,
			errorMsg:    "only supports db 0",
		},
		{
			name: "cluster without addresses",
			config: StoreConfig{
				Enabled:     true,
				TTLSeconds:  3600,
				BackendType: RedisStoreType,
				Redis: RedisStoreConfig{
					ClusterMode: true,
					DB:          0,
				},
			},
			expectError: true,
			errorMsg:    "cluster_addresses is empty",
		},
		{
			name: "standalone without address",
			config: StoreConfig{
				Enabled:     true,
				TTLSeconds:  3600,
				BackendType: RedisStoreType,
				Redis: RedisStoreConfig{
					ClusterMode: false,
					DB:          0,
				},
			},
			expectError: true,
			errorMsg:    "address is required",
		},
		{
			name: "invalid DB number",
			config: StoreConfig{
				Enabled:     true,
				TTLSeconds:  3600,
				BackendType: RedisStoreType,
				Redis: RedisStoreConfig{
					Address: "localhost:6379",
					DB:      20,
				},
			},
			expectError: true,
			errorMsg:    "invalid DB number",
		},
	}
}

// TestRedisStoreConfig tests configuration validation and defaults
func TestRedisStoreConfig(t *testing.T) {
	tests := buildRedisStoreConfigTests()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewRedisStore(tt.config)
			if tt.expectError {
				require.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else if err != nil {
				// Note: This will fail if Redis is not running
				// In a real unit test, we would mock the Redis client
				t.Skipf("Redis not available for testing: %v", err)
			}
		})
	}
}

// TestRedisStoreDefaults tests that defaults are applied correctly
func TestRedisStoreDefaults(t *testing.T) {
	cfg := RedisStoreConfig{
		Address: "localhost:6379",
		DB:      0,
	}

	applyRedisConfigDefaults(&cfg)

	assert.Equal(t, "sr:", cfg.KeyPrefix)
	assert.Equal(t, 10, cfg.PoolSize)
	assert.Equal(t, 2, cfg.MinIdleConns)
	assert.Equal(t, 3, cfg.MaxRetries)
	assert.Equal(t, 5, cfg.DialTimeout)
	assert.Equal(t, 3, cfg.ReadTimeout)
	assert.Equal(t, 3, cfg.WriteTimeout)
}

// TestRedisBuildKey tests key construction
func TestRedisBuildKey(t *testing.T) {
	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  3600,
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			Address:   "localhost:6379",
			DB:        0,
			KeyPrefix: "sr:",
		},
	}

	// Skip if Redis not available
	store, err := NewRedisStore(cfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
		return
	}
	defer store.Close()

	tests := []struct {
		suffix   string
		expected string
	}{
		{
			suffix:   "response:resp_123",
			expected: "sr:response:resp_123",
		},
		{
			suffix:   "conversation:conv_456",
			expected: "sr:conversation:conv_456",
		},
	}

	for _, tt := range tests {
		t.Run(tt.suffix, func(t *testing.T) {
			key := store.buildKey(tt.suffix)
			assert.Equal(t, tt.expected, key)
		})
	}
}

// TestRedisStoreValidation tests input validation
func TestRedisStoreValidation(t *testing.T) {
	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  3600,
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			Address: "localhost:6379",
			DB:      0,
		},
	}

	store, err := NewRedisStore(cfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
		return
	}
	defer store.Close()

	ctx := context.Background()

	t.Run("store nil response", func(t *testing.T) {
		err := store.StoreResponse(ctx, nil)
		assert.Error(t, err)
		assert.Equal(t, ErrInvalidInput, err)
	})

	t.Run("store response with empty ID", func(t *testing.T) {
		resp := &responseapi.StoredResponse{
			ID: "",
		}
		err := store.StoreResponse(ctx, resp)
		assert.Error(t, err)
		assert.Equal(t, ErrInvalidInput, err)
	})

	t.Run("get response with empty ID", func(t *testing.T) {
		_, err := store.GetResponse(ctx, "")
		assert.Error(t, err)
		assert.Equal(t, ErrInvalidInput, err)
	})

	t.Run("delete response with empty ID", func(t *testing.T) {
		err := store.DeleteResponse(ctx, "")
		assert.Error(t, err)
		assert.Equal(t, ErrInvalidInput, err)
	})

	t.Run("get non-existent response", func(t *testing.T) {
		_, err := store.GetResponse(ctx, "resp_nonexistent")
		assert.Equal(t, ErrNotFound, err)
	})

	t.Run("update non-existent response", func(t *testing.T) {
		resp := &responseapi.StoredResponse{
			ID: "resp_nonexistent",
		}
		err := store.UpdateResponse(ctx, resp)
		assert.Equal(t, ErrNotFound, err)
	})

	t.Run("delete non-existent response", func(t *testing.T) {
		err := store.DeleteResponse(ctx, "resp_nonexistent")
		assert.Equal(t, ErrNotFound, err)
	})
}

// TestRedisKeyPrefix tests custom key prefixes
func TestRedisKeyPrefix(t *testing.T) {
	tests := []struct {
		name           string
		configPrefix   string
		expectedPrefix string
	}{
		{
			name:           "default prefix",
			configPrefix:   "",
			expectedPrefix: "sr:",
		},
		{
			name:           "custom prefix with colon",
			configPrefix:   "myapp:responses:",
			expectedPrefix: "myapp:responses:",
		},
		{
			name:           "custom prefix without colon",
			configPrefix:   "test",
			expectedPrefix: "test:",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := StoreConfig{
				Enabled:     true,
				TTLSeconds:  3600,
				BackendType: RedisStoreType,
				Redis: RedisStoreConfig{
					Address:   "localhost:6379",
					DB:        0,
					KeyPrefix: tt.configPrefix,
				},
			}

			store, err := NewRedisStore(cfg)
			if err != nil {
				t.Skipf("Redis not available: %v", err)
				return
			}
			defer store.Close()

			assert.Equal(t, tt.expectedPrefix, store.keyPrefix)
		})
	}
}

// TestRedisStoreIsEnabled tests the IsEnabled method
func TestRedisStoreIsEnabled(t *testing.T) {
	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  3600,
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			Address: "localhost:6379",
			DB:      0,
		},
	}

	store, err := NewRedisStore(cfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
		return
	}
	defer store.Close()

	assert.True(t, store.IsEnabled())
}

// TestRedisStoreCheckConnection tests connection checking
func TestRedisStoreCheckConnection(t *testing.T) {
	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  3600,
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			Address: "localhost:6379",
			DB:      0,
		},
	}

	store, err := NewRedisStore(cfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
		return
	}
	defer store.Close()

	ctx := context.Background()
	err = store.CheckConnection(ctx)
	assert.NoError(t, err)
}

// TestConfigPathLoading tests external config file loading
func TestConfigPathLoading(t *testing.T) {
	// Create a temporary directory for config file
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "redis-config.yaml")

	// Create a test config YAML file
	configYAML := `address: redis.example.com:6380
db: 1
pool_size: 20
min_idle_conns: 5
max_retries: 5
dial_timeout: 10
read_timeout: 5
write_timeout: 5
key_prefix: "test:"`

	err := os.WriteFile(configPath, []byte(configYAML), 0o600)
	require.NoError(t, err)

	// Load config from file
	baseCfg := RedisStoreConfig{
		ConfigPath: configPath,
	}
	loadedCfg, err := loadRedisStoreConfig(baseCfg)
	require.NoError(t, err)

	// Verify loaded config values
	assert.Equal(t, "redis.example.com:6380", loadedCfg.Address)
	assert.Equal(t, 1, loadedCfg.DB)
	assert.Equal(t, 20, loadedCfg.PoolSize)
	assert.Equal(t, 5, loadedCfg.MinIdleConns)
	assert.Equal(t, 5, loadedCfg.MaxRetries)
	assert.Equal(t, 10, loadedCfg.DialTimeout)
	assert.Equal(t, 5, loadedCfg.ReadTimeout)
	assert.Equal(t, 5, loadedCfg.WriteTimeout)
	assert.Equal(t, "test:", loadedCfg.KeyPrefix)
}

// TestConfigPathLoadingError tests error handling for invalid config files
func TestConfigPathLoadingError(t *testing.T) {
	t.Run("non-existent file", func(t *testing.T) {
		cfg := RedisStoreConfig{
			ConfigPath: "/nonexistent/config.yaml",
		}
		_, err := loadRedisStoreConfig(cfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to read config file")
	})

	t.Run("invalid YAML syntax", func(t *testing.T) {
		tempDir := t.TempDir()
		configPath := filepath.Join(tempDir, "invalid.yaml")

		invalidYAML := `
  address: redis.example.com:6380
  invalid: [unclosed
  db: 1
`
		err := os.WriteFile(configPath, []byte(invalidYAML), 0o600)
		require.NoError(t, err)

		cfg := RedisStoreConfig{
			ConfigPath: configPath,
		}
		_, err = loadRedisStoreConfig(cfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to parse config file")
	})
}

// TestTLSConfig tests TLS configuration validation
func TestTLSConfig(t *testing.T) {
	t.Run("TLS enabled without cert paths", func(t *testing.T) {
		cfg := StoreConfig{
			Enabled:     true,
			TTLSeconds:  3600,
			BackendType: RedisStoreType,
			Redis: RedisStoreConfig{
				Address:    "localhost:6379",
				DB:         0,
				TLSEnabled: true,
				// Missing TLSCertPath and TLSKeyPath
			},
		}

		_, err := NewRedisStore(cfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "tls_cert_path")
	})

	t.Run("TLS enabled with non-existent cert", func(t *testing.T) {
		cfg := StoreConfig{
			Enabled:     true,
			TTLSeconds:  3600,
			BackendType: RedisStoreType,
			Redis: RedisStoreConfig{
				Address:     "localhost:6379",
				DB:          0,
				TLSEnabled:  true,
				TLSCertPath: "/nonexistent/cert.pem",
				TLSKeyPath:  "/nonexistent/key.pem",
			},
		}

		_, err := NewRedisStore(cfg)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})
}

// TestRedisConversationIndexKeyIsolation locks down the invariant that makes the
// secondary index safe to add to an existing keyspace: index keys must stay
// invisible to the scan patterns used elsewhere. ListConversations scans
// sr:conversation:* and unmarshals every hit as JSON, so a sorted set caught by
// that pattern would be read as a conversation blob.
//
// This needs no Redis, so it guards the invariant on every CI run.
func TestRedisConversationIndexKeyIsolation(t *testing.T) {
	store := &RedisStore{keyPrefix: "sr:"}

	indexKey := store.conversationIndexKey("conv_123")
	assert.Equal(t, "sr:conversation-index:conv_123", indexKey)

	// Both scan patterns are a literal prefix followed by "*", so being matched
	// by one is exactly the same as carrying its prefix.
	for _, scanPrefix := range []string{
		store.buildKey(ConversationKeyPrefix),
		store.buildKey(ResponseKeyPrefix),
	} {
		assert.Falsef(t, strings.HasPrefix(indexKey, scanPrefix),
			"index key %q must not be matched by the %q* scan pattern", indexKey, scanPrefix)
	}
}

// conversationIndexMembers returns the response IDs recorded in a conversation's
// secondary index, so tests can assert on the index itself rather than only on
// what a listing happens to return.
func conversationIndexMembers(t *testing.T, store *RedisStore, conversationID string) []string {
	t.Helper()

	members, err := store.client.ZRange(context.Background(), store.conversationIndexKey(conversationID), 0, -1).Result()
	require.NoError(t, err)
	return members
}

// newConversationIndexStore returns a store scoped to a key prefix unique to this
// run, so its keys cannot collide with the other suites sharing DB 0, and skips
// when no Redis is reachable (the convention used throughout this file).
func newConversationIndexStore(t *testing.T) *RedisStore {
	t.Helper()

	cfg := StoreConfig{
		Enabled:     true,
		TTLSeconds:  300,
		BackendType: RedisStoreType,
		Redis: RedisStoreConfig{
			Address:   "localhost:6379",
			DB:        0,
			KeyPrefix: fmt.Sprintf("srtest:%d:", time.Now().UnixNano()),
		},
	}

	store, err := NewRedisStore(cfg)
	if err != nil {
		t.Skipf("Redis not available: %v", err)
	}

	t.Cleanup(func() {
		ctx := context.Background()
		iter := store.client.Scan(ctx, 0, store.buildKey("*"), 0).Iterator()
		for iter.Next(ctx) {
			store.client.Del(ctx, iter.Val())
		}
		_ = store.Close()
	})

	return store
}

// TestRedisConversationIndexListing covers the index-backed read path: what it
// returns, in what order, and how it converges when the index and the payloads
// disagree.
func TestRedisConversationIndexListing(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	now := time.Now().Unix()
	storeResponse := func(t *testing.T, id, convID string, createdAt int64) {
		t.Helper()
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID:             id,
			ConversationID: convID,
			Status:         "completed",
			CreatedAt:      createdAt,
			Output:         []responseapi.OutputItem{{ID: "item_" + id}},
		}))
	}

	t.Run("returns only the requested conversation, oldest first", func(t *testing.T) {
		storeResponse(t, "resp_idx_a1", "conv_idx_a", now)
		storeResponse(t, "resp_idx_a2", "conv_idx_a", now+1)
		storeResponse(t, "resp_idx_b1", "conv_idx_b", now)
		// A response with no conversation must not land in any index.
		storeResponse(t, "resp_idx_orphan", "", now)

		responses, err := store.ListResponsesByConversation(ctx, "conv_idx_a", ListOptions{})
		require.NoError(t, err)
		require.Len(t, responses, 2)
		// The index is scored by created_at, so it reads back chronologically.
		assert.Equal(t, "resp_idx_a1", responses[0].ID)
		assert.Equal(t, "resp_idx_a2", responses[1].ID)

		assert.Equal(t, []string{"resp_idx_a1", "resp_idx_a2"}, conversationIndexMembers(t, store, "conv_idx_a"))
		assert.Equal(t, []string{"resp_idx_b1"}, conversationIndexMembers(t, store, "conv_idx_b"))
	})

	t.Run("unknown conversation returns nothing", func(t *testing.T) {
		responses, err := store.ListResponsesByConversation(ctx, "conv_idx_missing", ListOptions{})
		assert.NoError(t, err)
		assert.Empty(t, responses)
	})

	t.Run("empty conversation ID is rejected", func(t *testing.T) {
		_, err := store.ListResponsesByConversation(ctx, "", ListOptions{})
		assert.ErrorIs(t, err, ErrInvalidInput)
	})

	t.Run("deleting a response drops its index entry", func(t *testing.T) {
		require.NoError(t, store.DeleteResponse(ctx, "resp_idx_a1"))

		assert.Equal(t, []string{"resp_idx_a2"}, conversationIndexMembers(t, store, "conv_idx_a"))

		responses, err := store.ListResponsesByConversation(ctx, "conv_idx_a", ListOptions{})
		require.NoError(t, err)
		require.Len(t, responses, 1)
		assert.Equal(t, "resp_idx_a2", responses[0].ID)
	})

	t.Run("moving a response between conversations reindexes it", func(t *testing.T) {
		moved, err := store.GetResponse(ctx, "resp_idx_a2")
		require.NoError(t, err)

		moved.ConversationID = "conv_idx_b"
		require.NoError(t, store.UpdateResponse(ctx, moved))

		fromOld, err := store.ListResponsesByConversation(ctx, "conv_idx_a", ListOptions{})
		require.NoError(t, err)
		assert.Empty(t, fromOld)
		assert.Empty(t, conversationIndexMembers(t, store, "conv_idx_a"))

		toNew, err := store.ListResponsesByConversation(ctx, "conv_idx_b", ListOptions{})
		require.NoError(t, err)
		require.Len(t, toNew, 2)
		assert.Equal(t, []string{"resp_idx_b1", "resp_idx_a2"}, []string{toNew[0].ID, toNew[1].ID})
	})

	t.Run("listing prunes entries whose payload is gone", func(t *testing.T) {
		// Drop the payload behind the store's back, the way a TTL expiry would.
		require.NoError(t, store.client.Del(ctx, store.buildKey(ResponseKeyPrefix+"resp_idx_b1")).Err())

		responses, err := store.ListResponsesByConversation(ctx, "conv_idx_b", ListOptions{})
		require.NoError(t, err)
		require.Len(t, responses, 1)
		assert.Equal(t, "resp_idx_a2", responses[0].ID)

		assert.Equal(t, []string{"resp_idx_a2"}, conversationIndexMembers(t, store, "conv_idx_b"))
	})
}

// TestRedisDeleteConversationCascade covers a conversation holding more responses
// than one page: the cascade reads the index directly, so it must not stop at
// DefaultListLimit the way the previous list-driven implementation did.
func TestRedisDeleteConversationCascade(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	const (
		convID      = "conv_cascade"
		responseCnt = DefaultListLimit + 5
		otherConvID = "conv_cascade_other"
		otherRespID = "resp_cascade_other"
	)

	require.NoError(t, store.CreateConversation(ctx, &responseapi.StoredConversation{
		ID:        convID,
		CreatedAt: time.Now().Unix(),
	}))

	for i := 0; i < responseCnt; i++ {
		require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
			ID:             fmt.Sprintf("resp_cascade_%d", i),
			ConversationID: convID,
			Status:         "completed",
			CreatedAt:      time.Now().Unix() + int64(i),
			Output:         []responseapi.OutputItem{{ID: fmt.Sprintf("item_%d", i)}},
		}))
	}

	// A response in a different conversation must survive the cascade.
	require.NoError(t, store.StoreResponse(ctx, &responseapi.StoredResponse{
		ID:             otherRespID,
		ConversationID: otherConvID,
		Status:         "kept",
		CreatedAt:      time.Now().Unix(),
	}))

	require.NoError(t, store.DeleteConversation(ctx, convID, true))

	for i := 0; i < responseCnt; i++ {
		_, err := store.GetResponse(ctx, fmt.Sprintf("resp_cascade_%d", i))
		assert.ErrorIsf(t, err, ErrNotFound, "response %d should have been deleted with its conversation", i)
	}
	assert.Empty(t, conversationIndexMembers(t, store, convID))

	survivor, err := store.GetResponse(ctx, otherRespID)
	require.NoError(t, err)
	assert.Equal(t, "kept", survivor.Status)
}

// TestRedisStoreResponseRejectsDuplicate covers the SET NX path that both enforces
// the "must not already exist" contract and orders the payload write ahead of the
// index write.
func TestRedisStoreResponseRejectsDuplicate(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	response := &responseapi.StoredResponse{
		ID:             "resp_duplicate",
		ConversationID: "conv_duplicate",
		Status:         "completed",
		CreatedAt:      time.Now().Unix(),
	}
	require.NoError(t, store.StoreResponse(ctx, response))

	duplicate := *response
	duplicate.Status = "overwritten"
	assert.ErrorIs(t, store.StoreResponse(ctx, &duplicate), ErrAlreadyExists)

	// The original payload must be untouched, and indexed exactly once.
	stored, err := store.GetResponse(ctx, response.ID)
	require.NoError(t, err)
	assert.Equal(t, "completed", stored.Status)
	assert.Equal(t, []string{"resp_duplicate"}, conversationIndexMembers(t, store, "conv_duplicate"))
}
