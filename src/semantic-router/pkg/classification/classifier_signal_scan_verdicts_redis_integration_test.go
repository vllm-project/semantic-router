package classification

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net"
	"os"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// StorageIntegration: redis
func TestRedisSharedGuardVerdictsAcrossReplicas(t *testing.T) {
	storagetest.Require(t, "redis")
	host, port := os.Getenv("REDIS_HOST"), os.Getenv("REDIS_PORT")
	if host == "" {
		host = "127.0.0.1"
	}
	if port == "" {
		port = "6379"
	}
	address := net.JoinHostPort(host, port)
	admin := redis.NewClient(&redis.Options{Addr: address, Password: os.Getenv("REDIS_PASSWORD")})
	t.Cleanup(func() { _ = admin.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	if err := admin.Ping(ctx).Err(); err != nil {
		storagetest.Unavailable(t, "redis", err)
	}
	// The integration service may not require authentication. Give this test
	// its own authenticated ACL user; production never allows an empty secret.
	var secret [32]byte
	_, err := rand.Read(secret[:])
	require.NoError(t, err)
	password := hex.EncodeToString(secret[:])
	username := fmt.Sprintf("guard-test-%d", time.Now().UnixNano())
	require.NoError(t, admin.ACLSetUser(ctx, username, "on", ">"+password, "~vsr:guard-verdicts:v1:*", "+getrange", "+set", "+ping", "+hello", "+client").Err())
	t.Cleanup(func() { _ = admin.ACLDelUser(ctx, username).Err() })
	cfg := sharedVerdictTestConfig(t, address)
	// Exercise reuse independently of service scheduling under the race detector.
	// The production 50 ms budget intentionally falls back to scanning on delay.
	timeoutMs := 1000
	cfg.TimeoutMs = &timeoutMs
	t.Setenv("TEST_GUARD_REDIS_PASSWORD", password)
	cfg.Redis.Username = username
	configure := func(router *config.RouterConfig) { router.GuardVerdicts = cfg }
	first, firstModel := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond, configure)
	second, secondModel := refusingGuardClassifier(t, 20*time.Second, "", time.Millisecond, configure)
	t.Cleanup(func() {
		for _, classifier := range []*Classifier{first, second} {
			for key := range classifier.windowVerdicts.entries {
				_ = admin.Del(ctx, classifier.sharedVerdicts.key(key)).Err()
			}
		}
	})
	text := username + "\n" + scanProse(12_000)
	want := evaluateTurn(first, text)
	read := len(guardStates(firstModel))
	require.Greater(t, read, 2)
	for _, classifier := range []*Classifier{second, first, second} {
		got := evaluateTurn(classifier, text)
		require.Equal(t, want.SignalValues, got.SignalValues)
		require.Equal(t, want.ScanCoverage, got.ScanCoverage)
	}
	require.Len(t, guardStates(firstModel), read)
	require.Empty(t, guardStates(secondModel))

	// An unavailable authenticated connection still scans a fresh attack.
	require.NoError(t, second.sharedVerdicts.client.Close())
	attack := evaluateTurn(second, scanInjection+" "+text)
	require.True(t, guarded(attack))
	require.NotEmpty(t, guardStates(secondModel))
}
