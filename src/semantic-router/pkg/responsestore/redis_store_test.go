package responsestore

import (
	"context"
	"os"
	"path/filepath"
	"testing"

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
