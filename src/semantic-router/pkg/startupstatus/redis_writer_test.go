package startupstatus

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

func statusWriter(t *testing.T, addr, id string, ttl time.Duration) *RedisWriter {
	t.Helper()
	w, err := NewRedisWriter(RedisWriterConfig{Address: addr, InstanceID: id, TTL: ttl})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, w.Close()) })
	return w
}

func TestRedisStatusSeparatesProcessIncarnations(t *testing.T) {
	server := miniredis.RunT(t)
	first := statusWriter(t, server.Addr(), "first", time.Second)
	second := statusWriter(t, server.Addr(), "second", time.Second)
	restarted := statusWriter(t, server.Addr(), "restarted", time.Second)
	require.NoError(t, first.Write(State{Ready: true, Phase: "ready"}))
	require.NoError(t, second.Write(State{Phase: "loading"}))
	require.NoError(t, restarted.Write(State{Phase: "starting"}))
	for _, tc := range []struct {
		id, phase string
		ready     bool
	}{
		{"first", "ready", true}, {"second", "loading", false}, {"restarted", "starting", false},
	} {
		got, err := LoadFromRedis(first.client, tc.id)
		require.NoError(t, err)
		require.Equal(t, tc.id, got.InstanceID)
		require.Equal(t, tc.phase, got.Phase)
		require.Equal(t, tc.ready, got.Ready)
	}
	require.NoError(t, first.Close())
	server.FastForward(time.Second)
	_, err := LoadFromRedis(second.client, "first")
	require.ErrorIs(t, err, redis.Nil)
	require.NoError(t, second.Write(State{Ready: true, Phase: "ready"}))
	_, err = LoadFromRedis(second.client, "first")
	require.ErrorIs(t, err, redis.Nil, "a live peer must not resurrect a stopped process")
}

func TestRedisStatusHeartbeatRefreshesWithoutChangingObservation(t *testing.T) {
	server := miniredis.RunT(t)
	w := statusWriter(t, server.Addr(), "replica", 300*time.Millisecond)
	state := State{Ready: true, PendingModels: []string{"model"}, EmbeddingProvider: &EmbeddingProviderStatus{Mode: "remote"}}
	require.NoError(t, w.Write(state))
	original, err := LoadFromRedis(w.client, "replica")
	require.NoError(t, err)
	state.PendingModels[0] = "mutated"
	state.EmbeddingProvider.Mode = "mutated"
	server.FastForward(250 * time.Millisecond)
	require.Eventually(t, func() bool { return server.TTL(w.key) == w.ttl }, time.Second, 5*time.Millisecond)
	got, err := LoadFromRedis(w.client, "replica")
	require.NoError(t, err)
	require.Equal(t, original, got, "heartbeat must not change the observation or timestamp")
	require.NoError(t, w.Close())
	require.ErrorIs(t, w.Write(State{}), context.Canceled)
	server.FastForward(w.ttl)
	require.False(t, server.Exists(w.key))
}

func TestRedisStatusRetriesLatestObservationAfterOutage(t *testing.T) {
	server := miniredis.RunT(t)
	server.SetError("ERR unavailable")
	w := statusWriter(t, server.Addr(), "replica", 30*time.Millisecond)
	require.Error(t, w.Write(State{Phase: "loading"}))
	require.Error(t, w.Write(State{Phase: "ready", Ready: true}))
	server.SetError("")
	require.Eventually(t, func() bool {
		got, err := LoadFromRedis(w.client, "replica")
		return err == nil && got.Ready && got.Phase == "ready"
	}, time.Second, 5*time.Millisecond)
	server.SetError("ERR unavailable")
	server.FastForward(w.ttl)
	_, err := LoadFromRedis(w.client, "replica")
	require.Error(t, err)
	server.SetError("")
	require.Eventually(t, func() bool { return server.Exists(w.key) }, time.Second, 5*time.Millisecond)
}

func TestRedisStatusReaderRejectsUnscopedAndStaleRecords(t *testing.T) {
	server := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	data, err := json.Marshal(State{InstanceID: "replica", Ready: true})
	require.NoError(t, err)
	require.NoError(t, server.Set("vllm-sr:startup-status", string(data)))
	_, err = LoadFromRedis(client, "replica")
	require.ErrorIs(t, err, redis.Nil)
	key, err := redisStatusKey("replica")
	require.NoError(t, err)
	require.NoError(t, server.Set(key, string(data)))
	_, err = LoadFromRedis(client, "replica")
	require.ErrorIs(t, err, redis.Nil, "records without expiry are not live observations")
	require.NoError(t, server.Set(key, `{"instance_id":"neighbor","ready":true}`))
	server.SetTTL(key, time.Second)
	_, err = LoadFromRedis(client, "replica")
	require.ErrorContains(t, err, "identity mismatch")
	for _, id := range []string{"", "*", "neighbor:extra"} {
		_, err = LoadFromRedis(client, id)
		require.Error(t, err)
		_, err = NewRedisWriter(RedisWriterConfig{InstanceID: id})
		require.Error(t, err)
	}
	_, err = NewRedisWriter(RedisWriterConfig{InstanceID: "valid", TTL: -time.Second})
	require.Error(t, err)
}

func TestRedisStatusConcurrentWritesAndClose(t *testing.T) {
	server := miniredis.RunT(t)
	w := statusWriter(t, server.Addr(), "replica", 30*time.Millisecond)
	var writers sync.WaitGroup
	errors := make(chan error, 80)
	for range 8 {
		writers.Go(func() {
			for range 10 {
				errors <- w.Write(State{Phase: "loading"})
			}
		})
	}
	writers.Wait()
	for range 80 {
		require.NoError(t, <-errors)
	}
	require.NoError(t, w.Write(State{Phase: "ready", Ready: true}))
	got, err := LoadFromRedis(w.client, "replica")
	require.NoError(t, err)
	require.True(t, got.Ready)
	for range 4 {
		writers.Go(func() { errors <- w.Close() })
	}
	writers.Wait()
	for range 4 {
		require.NoError(t, <-errors)
	}
}

func TestRedisStatusRecoversWhenServerStartsAfterPublisher(t *testing.T) {
	server := miniredis.RunT(t)
	addr := server.Addr()
	server.Close()
	w := statusWriter(t, addr, "replica", 90*time.Millisecond)
	require.Error(t, w.Write(State{Ready: true, Phase: "ready"}))
	require.NoError(t, server.Restart())
	require.Eventually(t, func() bool {
		got, err := LoadFromRedis(w.client, "replica")
		return err == nil && got.Ready
	}, 5*time.Second, 20*time.Millisecond)
}
