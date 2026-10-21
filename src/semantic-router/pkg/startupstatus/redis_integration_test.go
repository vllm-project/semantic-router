package startupstatus

import (
	"crypto/rand"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

// Run against a real Redis/Valkey server to exercise wall-clock TTL and MULTI.
func TestRedisStatusLifecycleIntegration(t *testing.T) {
	addr := os.Getenv("STARTUP_STATUS_REDIS_TEST_ADDR")
	if addr == "" {
		t.Skip("set STARTUP_STATUS_REDIS_TEST_ADDR to run against Redis or Valkey")
	}
	firstID, secondID := rand.Text(), rand.Text()
	first := statusWriter(t, addr, firstID, 900*time.Millisecond)
	second := statusWriter(t, addr, secondID, 900*time.Millisecond)
	reader := redis.NewClient(&redis.Options{Addr: addr})
	t.Cleanup(func() { _ = reader.Close() })
	require.NoError(t, first.Write(State{Ready: true, Phase: "ready"}))
	require.NoError(t, second.Write(State{Phase: "loading"}))
	original, err := LoadFromRedis(reader, firstID)
	require.NoError(t, err)
	// The ready observation must survive several real TTL intervals unchanged.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		got, readErr := LoadFromRedis(reader, firstID)
		require.NoError(t, readErr)
		require.Equal(t, original, got)
		peer, peerErr := LoadFromRedis(reader, secondID)
		require.NoError(t, peerErr)
		require.False(t, peer.Ready)
		time.Sleep(50 * time.Millisecond)
	}
	require.NoError(t, first.Close())
	replacementID := rand.Text()
	replacement := statusWriter(t, addr, replacementID, 900*time.Millisecond)
	require.NoError(t, replacement.Write(State{Phase: "starting"}))
	require.Eventually(t, func() bool {
		_, readErr := LoadFromRedis(reader, firstID)
		return errors.Is(readErr, redis.Nil)
	}, 2*time.Second, 25*time.Millisecond)
	got, err := LoadFromRedis(reader, replacementID)
	require.NoError(t, err)
	require.False(t, got.Ready)
	peer, err := LoadFromRedis(reader, secondID)
	require.NoError(t, err)
	require.Equal(t, "loading", peer.Phase)
}
