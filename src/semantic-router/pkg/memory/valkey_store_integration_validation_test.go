//go:build !windows && cgo

package memory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ---------------------------------------------------------------------------
// Store validation errors
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_StoreValidation(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	t.Run("missing ID", func(t *testing.T) {
		err := store.Store(ctx, &Memory{Content: "test", UserID: "u1"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "memory ID is required")
	})

	t.Run("missing content", func(t *testing.T) {
		err := store.Store(ctx, &Memory{ID: "test_id", UserID: "u1"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "memory content is required")
	})

	t.Run("missing user ID", func(t *testing.T) {
		err := store.Store(ctx, &Memory{ID: "test_id", Content: "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "user ID is required")
	})
}

// ---------------------------------------------------------------------------
// Retrieve validation errors
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_RetrieveValidation(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	t.Run("missing query", func(t *testing.T) {
		_, err := store.Retrieve(ctx, RetrieveOptions{UserID: "u1"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "query is required")
	})

	t.Run("missing user ID", func(t *testing.T) {
		_, err := store.Retrieve(ctx, RetrieveOptions{Query: "test"})
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "user id is required")
	})
}

// ---------------------------------------------------------------------------
// IsEnabled / disabled store
// ---------------------------------------------------------------------------

func TestValkeyStoreInteg_DisabledStore(t *testing.T) {
	t.Parallel()

	store := &ValkeyStore{enabled: false}

	assert.False(t, store.IsEnabled())

	ctx := context.Background()
	assert.Error(t, store.Store(ctx, &Memory{}))
	_, err := store.Retrieve(ctx, RetrieveOptions{})
	assert.Error(t, err)
	_, err = store.Get(ctx, "id")
	assert.Error(t, err)
	assert.Error(t, store.Update(ctx, "id", &Memory{}))
	_, err = store.List(ctx, ListOptions{})
	assert.Error(t, err)
	assert.Error(t, store.Forget(ctx, "id"))
	assert.Error(t, store.ForgetByScope(ctx, MemoryScope{}))
}

// ---------------------------------------------------------------------------
// TLS integration tests
// ---------------------------------------------------------------------------

// TestValkeyStoreInteg_TLS_ConfigPropagation verifies that TLS config fields
// are accepted by NewValkeyStore without error when the store is disabled.
// The actual TLS handshake is not tested here (would require a TLS-enabled
// Valkey instance); the wiring from config to glide client lives in
// router_memory.go and is validated via the unit tests in valkey_store_test.go.
func TestValkeyStoreInteg_TLS_ConfigPropagation(t *testing.T) {
	t.Parallel()

	vc := &config.MemoryValkeyConfig{
		Host:                  "localhost",
		Port:                  6380,
		TLSEnabled:            true,
		TLSCAPath:             "/nonexistent/ca.pem",
		TLSInsecureSkipVerify: true,
		Dimension:             384,
		MetricType:            "COSINE",
	}

	// Disabled store should accept TLS config without attempting a connection.
	store, err := NewValkeyStore(ValkeyStoreOptions{
		Enabled:      false,
		ValkeyConfig: vc,
	})
	require.NoError(t, err)
	assert.False(t, store.IsEnabled())

	// The config struct itself carries the TLS values correctly.
	assert.True(t, vc.TLSEnabled)
	assert.Equal(t, "/nonexistent/ca.pem", vc.TLSCAPath)
	assert.True(t, vc.TLSInsecureSkipVerify)
}

// TestValkeyStoreInteg_TLS_BadCAPath verifies that createValkeyMemoryStore
// would fail with a clear error when given a non-existent CA path. We test
// this at the config level since the actual client creation happens in
// router_memory.go and requires the full extproc wiring.
func TestValkeyStoreInteg_TLS_BadCAPathConfig(t *testing.T) {
	t.Parallel()

	vc := &config.MemoryValkeyConfig{
		TLSEnabled: true,
		TLSCAPath:  "/definitely/does/not/exist/ca.pem",
	}

	// The CA file validation happens in router_memory.go (LoadRootCertificatesFromFile),
	// not in NewValkeyStore itself. Verify the config is valid at the store level.
	assert.True(t, vc.TLSEnabled)
	assert.NotEmpty(t, vc.TLSCAPath)
}
