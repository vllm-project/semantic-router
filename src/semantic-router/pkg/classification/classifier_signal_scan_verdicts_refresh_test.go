package classification

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func completeTestVerdict() modelservice.Response {
	return modelservice.Response{Answers: map[string]modelservice.Answer{"guard": {InputCoverage: "complete", Choice: "benign"}}}
}

func TestSharedVerdictRefreshesCoalesceAndPreserveLastUseExpiry(t *testing.T) {
	server, cfg := sharedVerdictTestServer(t)
	store := newVerdictTestStore(t, cfg, "recipe")
	pauseVerdictRefreshWorker(store)
	key, response := sha256.Sum256([]byte("window")), completeTestVerdict()
	var cache windowVerdicts
	cache.put(key, response)
	published := time.Now().Add(windowVerdictTTL - sharedVerdictRefreshInterval - time.Second)
	require.True(t, store.putUntil(context.Background(), key, response, published))
	cache.published(key, published)
	before := server.CommandCount()
	for range 100 {
		store.touch(&cache, key, response)
	}
	require.Equal(t, before, server.CommandCount(), "local hits must not write Redis")
	store.refreshMu.Lock()
	require.Len(t, store.refreshes, 1)
	lastExpiry := store.refreshes[store.key(key)].expires
	store.refreshMu.Unlock()
	require.False(t, store.flushRefreshes())
	require.Equal(t, before+1, server.CommandCount(), "one trailing write publishes all the hits")
	data, err := server.Get(store.key(key))
	require.NoError(t, err)
	var signed signedWindowVerdict
	var entry sharedWindowVerdict
	require.NoError(t, json.Unmarshal([]byte(data), &signed))
	require.NoError(t, json.Unmarshal(signed.Payload, &entry))
	require.Equal(t, lastExpiry.UnixMilli(), entry.ExpiresAt, "queue delay must not extend retention")
	require.Equal(t, lastExpiry, cache.sharedExpiry(key))
}

type heldVerdictWrite struct {
	hold    atomic.Bool
	entered chan struct{}
	release chan struct{}
}

func (h *heldVerdictWrite) DialHook(next redis.DialHook) redis.DialHook { return next }
func (h *heldVerdictWrite) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *heldVerdictWrite) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
	return func(ctx context.Context, cmd redis.Cmder) error {
		if cmd.Name() == "set" && h.hold.Load() {
			select {
			case h.entered <- struct{}{}:
			default:
			}
			select {
			case <-h.release:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return next(ctx, cmd)
	}
}

func TestSharedVerdictLocalHitDoesNotWaitForRenewal(t *testing.T) {
	_, cfg := sharedVerdictTestServer(t)
	timeout := 1000
	cfg.TimeoutMs = &timeout
	store := newVerdictTestStore(t, cfg, "recipe")
	hook := &heldVerdictWrite{entered: make(chan struct{}, 1), release: make(chan struct{})}
	store.client.AddHook(hook)
	hook.hold.Store(true)
	defer close(hook.release)
	key, response := windowVerdictKey(sha256.Sum256([]byte("window"))), completeTestVerdict()
	c := &Classifier{sharedVerdicts: store}
	c.windowVerdicts.put(key, response)
	store.touch(&c.windowVerdicts, key, response)
	select {
	case <-hook.entered:
	case <-time.After(time.Second):
		t.Fatal("background renewal did not start")
	}
	call := &systemOneCall{verdict: &key, scan: &guardScan{reused: make([]bool, 1)}}
	finished := make(chan bool, 1)
	go func() { _, kept := c.keptWindow(context.Background(), call); finished <- kept }()
	select {
	case kept := <-finished:
		require.True(t, kept)
	case <-time.After(200 * time.Millisecond):
		t.Fatal("local hit waited for a blocked Redis write")
	}
}

func TestSharedVerdictRefreshQueueIsBounded(t *testing.T) {
	_, cfg := sharedVerdictTestServer(t)
	store := newVerdictTestStore(t, cfg, "recipe")
	pauseVerdictRefreshWorker(store)
	var cache windowVerdicts
	response := completeTestVerdict()
	full := metrics.GuardVerdictStoreOpsTotal.WithLabelValues("refresh", "queue_full")
	before := testutil.ToFloat64(full)
	for i := range sharedVerdictRefreshLimit + 1 {
		key := sha256.Sum256([]byte(fmt.Sprint(i)))
		cache.put(key, response)
		cache.published(key, time.Now().Add(time.Minute))
		store.touch(&cache, key, response)
	}
	store.refreshMu.Lock()
	require.Len(t, store.refreshes, sharedVerdictRefreshLimit)
	store.refreshMu.Unlock()
	require.Equal(t, before+1, testutil.ToFloat64(full))
}

// Queue unit tests drive the worker synchronously; the local-hit test above
// exercises the live goroutine and its I/O boundary.
func pauseVerdictRefreshWorker(store *sharedWindowVerdicts) {
	store.cancel()
	store.worker.Wait()
	store.ctx, store.cancel = context.WithCancel(context.Background())
}

func TestSharedVerdictCapacityWaitDoesNotTripCooldown(t *testing.T) {
	_, cfg := sharedVerdictTestServer(t)
	store := newVerdictTestStore(t, cfg, "recipe")
	for range sharedVerdictConcurrency {
		store.slots <- struct{}{}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	_, _, hit := store.get(ctx, windowVerdictKey{})
	require.False(t, hit)
	require.Zero(t, store.retryAfter.Load())
	for range sharedVerdictConcurrency {
		<-store.slots
	}
	require.NoError(t, store.client.Ping(context.Background()).Err())
}

func TestSharedVerdictInvalidEntriesAreObservableAndWarningsThrottled(t *testing.T) {
	server, cfg := sharedVerdictTestServer(t)
	store := newVerdictTestStore(t, cfg, "recipe")
	key := windowVerdictKey{}
	invalid := metrics.GuardVerdictStoreOpsTotal.WithLabelValues("get", "invalid")
	before := testutil.ToFloat64(invalid)
	require.NoError(t, server.Set(store.key(key), "forged"))
	_, _, hit := store.get(context.Background(), key)
	require.False(t, hit)
	next := store.invalidWarning.Load()
	require.Greater(t, next, time.Now().UnixNano())
	_, _, hit = store.get(context.Background(), key)
	require.False(t, hit)
	require.Equal(t, next, store.invalidWarning.Load())
	require.Equal(t, before+2, testutil.ToFloat64(invalid))
}

func TestSharedVerdictFreshHitsLeaveRoomForDueRenewals(t *testing.T) {
	store := &sharedWindowVerdicts{sharedVerdictStore: &sharedVerdictStore{
		ctx: context.Background(), refreshes: make(map[string]*sharedVerdictRefresh), wake: make(chan struct{}, 1),
	}}
	var cache windowVerdicts
	response := completeTestVerdict()
	for i := range sharedVerdictRefreshLimit + 1 {
		key := sha256.Sum256([]byte(fmt.Sprint(i)))
		cache.put(key, response)
		cache.published(key, time.Now().Add(windowVerdictTTL))
		store.touch(&cache, key, response)
	}
	require.Empty(t, store.refreshes, "fresh hits must not reserve the queue for five minutes")
	key := sha256.Sum256([]byte("due"))
	cache.put(key, response)
	cache.published(key, time.Now().Add(time.Minute))
	store.touch(&cache, key, response)
	require.Contains(t, store.refreshes, store.key(key), "a due renewal must remain eligible after many fresh hits")
}

func TestSharedVerdictWorkerDrainsReadyQueue(t *testing.T) {
	server, cfg := sharedVerdictTestServer(t)
	store := newVerdictTestStore(t, cfg, "recipe")
	pauseVerdictRefreshWorker(store)
	var cache windowVerdicts
	response := completeTestVerdict()
	for i := range 100 {
		key := sha256.Sum256([]byte(fmt.Sprint(i)))
		cache.put(key, response)
		cache.published(key, time.Now().Add(time.Minute))
		store.touch(&cache, key, response)
	}
	store.worker.Add(1)
	go store.refreshLoop()
	require.Eventually(t, func() bool { return len(server.Keys()) == 100 }, 3*time.Second, time.Millisecond,
		"one wake must drain every ready batch without polling or another hit")
}

func TestSharedVerdictInFlightRenewalDoesNotRepeatTinyExtension(t *testing.T) {
	server, cfg := sharedVerdictTestServer(t)
	store := newVerdictTestStore(t, cfg, "recipe")
	pauseVerdictRefreshWorker(store)
	key, response := sha256.Sum256([]byte("window")), completeTestVerdict()
	var cache windowVerdicts
	cache.put(key, response)
	published := time.Now().Add(time.Minute)
	require.True(t, store.putUntil(context.Background(), key, response, published))
	cache.published(key, published)
	before := server.CommandCount()
	hook := &heldVerdictWrite{entered: make(chan struct{}, 1), release: make(chan struct{})}
	store.client.AddHook(hook)
	hook.hold.Store(true)
	store.touch(&cache, key, response)
	finished := make(chan struct{})
	go func() { store.flushRefreshes(); close(finished) }()
	select {
	case <-hook.entered:
	case <-time.After(time.Second):
		close(hook.release)
		t.Fatal("renewal did not start")
	}
	store.touch(&cache, key, response)
	close(hook.release)
	<-finished
	require.False(t, store.flushRefreshes())
	require.Equal(t, before+1, server.CommandCount(), "a hit during publication must not cause a tiny second renewal")
}
