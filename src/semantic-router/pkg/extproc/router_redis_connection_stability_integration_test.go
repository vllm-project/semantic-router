//go:build integration

package extproc

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestBuildRouterComponentsRepeatedReloadsHaveStableRedisConnectionCount is
// the connection half of the repeated-reload stability ask that
// TestBuildRouterComponentsRepeatedReloadsHaveStableFileDescriptorCount can't
// cover without a live backend. Requires a reachable Redis (default
// localhost:6379, override with REDIS_ADDR); skipped like the rest of this
// package's Redis integration tests via SKIP_REDIS_TESTS=true.
func TestBuildRouterComponentsRepeatedReloadsHaveStableRedisConnectionCount(t *testing.T) {
	if os.Getenv("SKIP_REDIS_TESTS") == "true" {
		t.Skip("Redis integration tests skipped due to SKIP_REDIS_TESTS=true")
	}
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		addr = "localhost:6379"
	}

	admin := redis.NewClient(&redis.Options{Addr: addr})
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.Ping(context.Background()).Err(); err != nil {
		t.Skipf("Redis not reachable at %s: %v", addr, err)
	}

	cfg := &config.RouterConfig{
		ResponseAPI: config.ResponseAPIConfig{
			Enabled:      true,
			StoreBackend: "redis",
			TTLSeconds:   60,
			Redis: config.ResponseAPIRedisConfig{
				Address:   addr,
				KeyPrefix: "sr-test-conn-stability:",
			},
		},
	}

	baseline := stableConnectedClients(t, admin)

	const iterations = 30
	for i := 0; i < iterations; i++ {
		components, err := buildRouterComponents(cfg)
		require.NoError(t, err)
		router := components.buildRouter()
		require.NoError(t, router.Close())
	}

	requireConnectedClientsSettleTo(t, admin, baseline,
		"buildRouterComponents+Close leaked Redis connections across %d repeated reload cycles", iterations)
}

// connectedClients parses Redis's own view of how many clients are
// connected, from INFO clients' connected_clients line. This counts every
// client talking to the server, including admin, so a fixed offset (the
// baseline) rather than an absolute value is what a test compares against.
func connectedClients(t *testing.T, admin *redis.Client) int {
	t.Helper()
	info, err := admin.Info(context.Background(), "clients").Result()
	require.NoError(t, err)
	for _, line := range strings.Split(info, "\r\n") {
		if v, ok := strings.CutPrefix(line, "connected_clients:"); ok {
			n, err := strconv.Atoi(strings.TrimSpace(v))
			require.NoError(t, err)
			return n
		}
	}
	t.Fatal("connected_clients not found in INFO clients output")
	return 0
}

// stableConnectedClients mirrors stableGoroutineCount: it polls for
// quiescence, so a connection from an earlier test that is still closing
// doesn't get mistaken for this test's baseline.
func stableConnectedClients(t *testing.T, admin *redis.Client) int {
	t.Helper()
	var last int
	consecutive := 0
	require.Eventually(t, func() bool {
		current := connectedClients(t, admin)
		if current == last {
			consecutive++
		} else {
			consecutive = 0
			last = current
		}
		return consecutive >= 3
	}, 10*time.Second, 50*time.Millisecond, "connected_clients count never settled")
	return last
}

func requireConnectedClientsSettleTo(t *testing.T, admin *redis.Client, baseline int, msg string, args ...interface{}) {
	t.Helper()
	require.Eventuallyf(t, func() bool {
		return connectedClients(t, admin) <= baseline
	}, 10*time.Second, 50*time.Millisecond, msg, args...)
}
