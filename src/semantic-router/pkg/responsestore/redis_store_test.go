package responsestore

import (
	"context"
	"encoding/json"
	"errors"
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

// TestRedisConversationIndexKeyIsolation guards the invariant that index,
// empty-marker, and lock keys are invisible to the sr:conversation:* and
// sr:response:* scans in ListConversations/legacy backfill, which would
// otherwise read a sorted set or marker string as conversation/response
// JSON. Needs no Redis.
func TestRedisConversationIndexKeyIsolation(t *testing.T) {
	store := &RedisStore{keyPrefix: "sr:"}

	indexKey := store.conversationIndexKey("conv_123")
	assert.Equal(t, "sr:conversation-index:conv_123", indexKey)

	emptyMarkerKey := store.emptyConversationIndexMarkerKey("conv_123")
	assert.Equal(t, "sr:conversation-index-empty:conv_123", emptyMarkerKey)

	lockKey := store.conversationIndexLockKey("conv_123")
	assert.Equal(t, "sr:conversation-index-lock:conv_123", lockKey)

	// Each scan pattern is a literal prefix plus "*", so matching == having it.
	scanPrefixes := []string{
		store.buildKey(ConversationKeyPrefix),
		store.buildKey(ResponseKeyPrefix),
	}
	for _, key := range []string{indexKey, emptyMarkerKey, lockKey} {
		for _, scanPrefix := range scanPrefixes {
			assert.Falsef(t, strings.HasPrefix(key, scanPrefix),
				"key %q must not be matched by the %q* scan pattern", key, scanPrefix)
		}
	}

	// The empty marker and lock key families must also be distinct from the
	// index key family itself: either one accidentally matching the index
	// scan prefix would let a marker or lock be read back as sorted-set data
	// by anything that scans "conversation-index:*" (e.g. a future admin tool).
	indexScanPrefix := store.buildKey(ConversationIndexKeyPrefix)
	require.True(t, strings.HasPrefix(indexKey, indexScanPrefix))
	assert.Falsef(t, strings.HasPrefix(emptyMarkerKey, indexScanPrefix),
		"empty marker key %q must not be matched by the %q* index scan pattern", emptyMarkerKey, indexScanPrefix)
	assert.Falsef(t, strings.HasPrefix(lockKey, indexScanPrefix),
		"lock key %q must not be matched by the %q* index scan pattern", lockKey, indexScanPrefix)
}

// conversationIndexMembers reads the index directly, so tests can assert on it
// and not only on what a listing happens to return.
func conversationIndexMembers(t *testing.T, store *RedisStore, conversationID string) []string {
	t.Helper()

	members, err := store.client.ZRange(context.Background(), store.conversationIndexKey(conversationID), 0, -1).Result()
	require.NoError(t, err)
	return members
}

// newConversationIndexStore scopes the store to a key prefix unique to this run,
// so it cannot collide with the other suites sharing DB 0. Skips without Redis.
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
// returns, in what order, and how it converges when index and payloads disagree.
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

// TestRedisDeleteConversationCascade uses more responses than one page: the
// cascade reads the index directly, so it must not stop at DefaultListLimit.
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

// TestRedisStoreResponseRejectsDuplicate covers the SET NX path: it enforces the
// no-duplicate contract and orders the payload write ahead of the index write.
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

// directSetResponsePayload writes a response payload straight to Redis,
// bypassing StoreResponse and its indexing entirely — the same shape as data
// written before this index existed, or as an in-flight writer that landed
// its payload but has not indexed it yet.
func directSetResponsePayload(t *testing.T, store *RedisStore, response *responseapi.StoredResponse) {
	t.Helper()

	data, err := json.Marshal(response)
	require.NoError(t, err)

	key := store.buildKey(ResponseKeyPrefix + response.ID)
	require.NoError(t, store.client.Set(context.Background(), key, data, store.ttl).Err())
}

// TestRedisStoreResponseRollsBackIndexFailure covers blueprint §2.3/§3.5: if
// the payload write succeeds but indexing fails, StoreResponse must roll the
// payload back via compare-delete rather than leave an orphan that a retry
// can never repair (SETNX would just see it exists and hit ErrAlreadyExists
// with no index to fix).
func TestRedisStoreResponseRollsBackIndexFailure(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	injectedErr := errors.New("injected index failure")
	store.indexResponseOverride = func(context.Context, string, string, int64) error {
		return injectedErr
	}

	response := &responseapi.StoredResponse{
		ID:             "resp_rollback",
		ConversationID: "conv_rollback",
		Status:         "completed",
		CreatedAt:      time.Now().Unix(),
	}

	err := store.StoreResponse(ctx, response)
	require.Error(t, err)
	assert.ErrorIs(t, err, injectedErr)

	// The rolled-back payload must be gone, not orphaned: GetResponse sees a
	// clean slate, and a retry does not hit ErrAlreadyExists.
	_, err = store.GetResponse(ctx, response.ID)
	assert.ErrorIs(t, err, ErrNotFound)
	assert.Empty(t, conversationIndexMembers(t, store, "conv_rollback"))

	// Remove the injected failure and retry: must succeed cleanly.
	store.indexResponseOverride = nil
	require.NoError(t, store.StoreResponse(ctx, response))

	stored, err := store.GetResponse(ctx, response.ID)
	require.NoError(t, err)
	assert.Equal(t, "completed", stored.Status)
	assert.Equal(t, []string{"resp_rollback"}, conversationIndexMembers(t, store, "conv_rollback"))
}

// TestRedisCompareDeleteResponsePayload covers the ABA race the blueprint
// calls out (§2.3): rollback must never delete a payload that changed after
// the failed write, e.g. because a concurrent writer replaced it once the
// original value's TTL expired.
func TestRedisCompareDeleteResponsePayload(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	key := store.buildKey(ResponseKeyPrefix + "resp_cas")

	t.Run("deletes when value matches", func(t *testing.T) {
		require.NoError(t, store.client.Set(ctx, key, []byte("payload-a"), store.ttl).Err())

		deleted, err := store.compareDeleteResponsePayload(ctx, key, []byte("payload-a"))
		require.NoError(t, err)
		assert.True(t, deleted)

		exists, err := store.client.Exists(ctx, key).Result()
		require.NoError(t, err)
		assert.Zero(t, exists)
	})

	t.Run("leaves a changed value untouched", func(t *testing.T) {
		require.NoError(t, store.client.Set(ctx, key, []byte("payload-a"), store.ttl).Err())

		deleted, err := store.compareDeleteResponsePayload(ctx, key, []byte("payload-b"))
		require.NoError(t, err)
		assert.False(t, deleted)

		value, err := store.client.Get(ctx, key).Result()
		require.NoError(t, err)
		assert.Equal(t, "payload-a", value)
	})

	t.Run("no-op against a missing key", func(t *testing.T) {
		require.NoError(t, store.client.Del(ctx, key).Err())

		deleted, err := store.compareDeleteResponsePayload(ctx, key, []byte("anything"))
		require.NoError(t, err)
		assert.False(t, deleted)
	})
}

// TestRedisStoreResponseDuplicateRepairsOrphanedIndex covers the repair half
// of blueprint §3.5: a retry that lands on an existing payload whose index
// entry is missing (e.g. left behind by a prior partial failure, or never
// indexed pre-upgrade) must repair the index from the stored payload before
// returning the duplicate error.
func TestRedisStoreResponseDuplicateRepairsOrphanedIndex(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	response := &responseapi.StoredResponse{
		ID:             "resp_orphan",
		ConversationID: "conv_orphan",
		Status:         "completed",
		CreatedAt:      time.Now().Unix(),
	}
	directSetResponsePayload(t, store, response)
	require.Empty(t, conversationIndexMembers(t, store, "conv_orphan"))

	retry := *response
	err := store.StoreResponse(ctx, &retry)
	assert.ErrorIs(t, err, ErrAlreadyExists)

	assert.Equal(t, []string{"resp_orphan"}, conversationIndexMembers(t, store, "conv_orphan"))
}

// TestRedisStoreResponseDuplicateDifferentConversationNotRepaired covers the
// poisoning guard from blueprint §2.3/§3.5: a duplicate ID whose stored
// payload belongs to a different conversation than the retry attempted must
// not be indexed under the attempted conversation.
func TestRedisStoreResponseDuplicateDifferentConversationNotRepaired(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	stored := &responseapi.StoredResponse{
		ID:             "resp_mismatch",
		ConversationID: "conv_real",
		Status:         "completed",
		CreatedAt:      time.Now().Unix(),
	}
	directSetResponsePayload(t, store, stored)

	attempted := *stored
	attempted.ConversationID = "conv_wrong"
	err := store.StoreResponse(ctx, &attempted)
	assert.ErrorIs(t, err, ErrAlreadyExists)

	assert.Empty(t, conversationIndexMembers(t, store, "conv_wrong"))
	// The real conversation was never indexed either: repair only runs
	// against the attempted conversation, and stored != attempted here.
	assert.Empty(t, conversationIndexMembers(t, store, "conv_real"))
}

// TestRedisStoreResponseDuplicateNoConversationNotRepaired covers the "no
// index expected" branch of repairExistingResponseIndex: a stored response
// with no ConversationID has nothing to repair.
func TestRedisStoreResponseDuplicateNoConversationNotRepaired(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	stored := &responseapi.StoredResponse{
		ID:        "resp_no_conv",
		Status:    "completed",
		CreatedAt: time.Now().Unix(),
	}
	directSetResponsePayload(t, store, stored)

	attempted := *stored
	attempted.ConversationID = "conv_should_not_exist"
	err := store.StoreResponse(ctx, &attempted)
	assert.ErrorIs(t, err, ErrAlreadyExists)

	assert.Empty(t, conversationIndexMembers(t, store, "conv_should_not_exist"))
}

// TestRedisListResponsesByConversationLazyBackfill covers the upgrade path
// (blueprint §6.2): responses written before the index existed — simulated
// with directSetResponsePayload, which bypasses StoreResponse's indexing —
// must still be discoverable on first read, get backfilled into the index,
// and must not trigger a second O(N) scan on the next read of the same
// conversation.
func TestRedisListResponsesByConversationLazyBackfill(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	now := time.Now().Unix()
	legacy := []*responseapi.StoredResponse{
		{ID: "resp_legacy_1", ConversationID: "conv_legacy", Status: "completed", CreatedAt: now},
		{ID: "resp_legacy_2", ConversationID: "conv_legacy", Status: "completed", CreatedAt: now + 1},
		{ID: "resp_legacy_3", ConversationID: "conv_legacy", Status: "completed", CreatedAt: now + 2},
	}
	for _, response := range legacy {
		directSetResponsePayload(t, store, response)
	}
	require.Empty(t, conversationIndexMembers(t, store, "conv_legacy"),
		"precondition: no index should exist yet for legacy data")

	responses, err := store.ListResponsesByConversation(ctx, "conv_legacy", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 3)
	assert.Equal(t, []string{"resp_legacy_1", "resp_legacy_2", "resp_legacy_3"},
		[]string{responses[0].ID, responses[1].ID, responses[2].ID})
	assert.Equal(t, 1, store.scanInvocations, "first read must run exactly one legacy scan")

	assert.ElementsMatch(t, []string{"resp_legacy_1", "resp_legacy_2", "resp_legacy_3"},
		conversationIndexMembers(t, store, "conv_legacy"))
	assert.Zero(t, exists(t, store, store.emptyConversationIndexMarkerKey("conv_legacy")))

	// The index now exists, so a second read must go straight through it
	// without scanning the keyspace again.
	responses, err = store.ListResponsesByConversation(ctx, "conv_legacy", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 3)
	assert.Equal(t, 1, store.scanInvocations, "second read of an already-backfilled conversation must not scan again")
}

// exists reports whether a raw Redis key exists, for asserting on marker/
// index presence directly rather than only through store methods.
func exists(t *testing.T, store *RedisStore, key string) int64 {
	t.Helper()
	n, err := store.client.Exists(context.Background(), key).Result()
	require.NoError(t, err)
	return n
}

// TestRedisListResponsesByConversationEmptyMarker covers blueprint §2.4/§3.4:
// a legitimately empty or unknown conversation must not force a full
// keyspace scan on every read — only once, after which the empty marker
// short-circuits subsequent reads.
func TestRedisListResponsesByConversationEmptyMarker(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	// Seed unrelated data so the scan has something to walk past.
	directSetResponsePayload(t, store, &responseapi.StoredResponse{
		ID: "resp_unrelated", ConversationID: "conv_other", Status: "completed", CreatedAt: time.Now().Unix(),
	})

	responses, err := store.ListResponsesByConversation(ctx, "conv_never_existed", ListOptions{})
	require.NoError(t, err)
	assert.Empty(t, responses)
	assert.Equal(t, 1, store.scanInvocations)

	markerKey := store.emptyConversationIndexMarkerKey("conv_never_existed")
	assert.EqualValues(t, 1, exists(t, store, markerKey))
	ttl, err := store.client.TTL(ctx, markerKey).Result()
	require.NoError(t, err)
	assert.Positive(t, ttl, "empty marker must carry a positive TTL, not be immortal")

	// Repeated reads of the same empty conversation must not scan again.
	for i := 0; i < 3; i++ {
		responses, err = store.ListResponsesByConversation(ctx, "conv_never_existed", ListOptions{})
		require.NoError(t, err)
		assert.Empty(t, responses)
	}
	assert.Equal(t, 1, store.scanInvocations, "repeated reads of an empty conversation must not force repeated scans")
}

// TestRedisLazyBackfillConcurrentWriteNotHidden covers blueprint §2.2: a
// response indexed normally by a concurrent StoreResponse call, landing
// while a lazy legacy scan for the same conversation is in flight, must
// survive — the scan's own findings only ever add to the index, and
// indexResponse always clears the empty marker, so neither ordering can make
// the concurrent write disappear behind a "confirmed empty" marker.
func TestRedisLazyBackfillConcurrentWriteNotHidden(t *testing.T) {
	store := newConversationIndexStore(t)
	ctx := context.Background()

	legacy := &responseapi.StoredResponse{
		ID: "resp_race_legacy", ConversationID: "conv_race", Status: "completed", CreatedAt: time.Now().Unix(),
	}
	directSetResponsePayload(t, store, legacy)

	concurrent := &responseapi.StoredResponse{
		ID: "resp_race_concurrent", ConversationID: "conv_race", Status: "completed", CreatedAt: time.Now().Unix() + 1,
	}
	store.lazyBackfillPreScanHook = func() {
		// Runs once, before the scan walks the keyspace: lands a normal
		// indexed write for the same conversation the backfill is about to
		// scan for, so the scan observes both the legacy and the
		// concurrently-indexed payload.
		require.NoError(t, store.StoreResponse(ctx, concurrent))
	}

	responses, err := store.ListResponsesByConversation(ctx, "conv_race", ListOptions{Order: "asc"})
	require.NoError(t, err)
	require.Len(t, responses, 2)
	assert.Equal(t, []string{"resp_race_legacy", "resp_race_concurrent"},
		[]string{responses[0].ID, responses[1].ID})

	assert.ElementsMatch(t, []string{"resp_race_legacy", "resp_race_concurrent"},
		conversationIndexMembers(t, store, "conv_race"))
	assert.Zero(t, exists(t, store, store.emptyConversationIndexMarkerKey("conv_race")),
		"the concurrently indexed write must clear any empty marker, not be hidden behind one")
}
