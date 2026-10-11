package memory

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/require"
)

type unimplementedStore = Store

// fenceStore serves one memory until Forget. With stale set it keeps serving
// it afterwards, like a Bounded-consistency read. gate pauses the first Retrieve
// after it has read, so a test can delete while that retrieval is in flight.
type fenceStore struct {
	unimplementedStore
	mu      sync.Mutex
	mem     *Memory
	deleted bool
	stale   bool
	calls   int
	read    chan struct{}
	gate    chan struct{}
}

func (s *fenceStore) Retrieve(context.Context, RetrieveOptions) ([]*RetrieveResult, error) {
	s.mu.Lock()
	s.calls++
	var results []*RetrieveResult
	if !s.deleted || s.stale {
		results = []*RetrieveResult{{Memory: s.mem, Score: 1}}
	}
	read, gate := s.read, s.gate
	s.read, s.gate = nil, nil
	s.mu.Unlock()
	if read != nil {
		close(read)
		<-gate
	}
	return results, nil
}

func (s *fenceStore) Get(context.Context, string) (*Memory, error) {
	return s.mem, nil
}

func (s *fenceStore) Forget(context.Context, string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.deleted = true
	return nil
}

func (s *fenceStore) backendCalls() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

func newFenceFixture(t *testing.T, store *fenceStore) (*miniredis.Miniredis, *CachingStore, *RedisCache) {
	t.Helper()
	mr := miniredis.RunT(t)
	cache, err := NewRedisCache(context.Background(), &RedisCacheConfig{Address: mr.Addr(), TTLSeconds: 300})
	require.NoError(t, err)
	t.Cleanup(func() { _ = cache.Close() })
	return mr, NewCachingStore(store, cache, "test").(*CachingStore), cache
}

func TestCachingStoreRetrievalRacingForgetIsNotRecached(t *testing.T) {
	ctx := context.Background()
	read, gate := make(chan struct{}), make(chan struct{})
	store := &fenceStore{
		mem:  &Memory{ID: "secret", UserID: "u", Content: "allergic to peanuts"},
		read: read,
		gate: gate,
	}
	mr, cached, _ := newFenceFixture(t, store)
	opts := RetrieveOptions{Query: "allergies", UserID: "u", Limit: 5}

	inFlight := make(chan []*RetrieveResult)
	go func() {
		results, _ := cached.Retrieve(ctx, opts)
		inFlight <- results
	}()
	<-read
	require.NoError(t, cached.Forget(ctx, "secret"))
	close(gate)
	require.Len(t, <-inFlight, 1, "the in-flight retrieval read before the delete")

	mr.FastForward(cacheWriteHold + time.Second)
	results, err := cached.Retrieve(ctx, opts)
	require.NoError(t, err)
	require.Empty(t, results, "a forgotten memory must not be served from the cache")
	require.Equal(t, 2, store.backendCalls())
}

func TestCachingStoreDoesNotCacheStaleReadsDuringHold(t *testing.T) {
	ctx := context.Background()
	store := &fenceStore{mem: &Memory{ID: "secret", UserID: "u"}, stale: true}
	mr, cached, _ := newFenceFixture(t, store)
	opts := RetrieveOptions{Query: "q", UserID: "u", Limit: 5}

	require.NoError(t, cached.Forget(ctx, "secret"))
	for i := 0; i < 2; i++ {
		_, err := cached.Retrieve(ctx, opts)
		require.NoError(t, err)
	}
	require.Equal(t, 2, store.backendCalls(), "reads during the hold go to the backend and are not cached")

	mr.FastForward(cacheWriteHold + time.Second)
	for i := 0; i < 2; i++ {
		_, err := cached.Retrieve(ctx, opts)
		require.NoError(t, err)
	}
	require.Equal(t, 3, store.backendCalls(), "caching resumes once the hold expires")
}

func TestRedisCacheInvalidationReplacesGenerationAndHolds(t *testing.T) {
	ctx := context.Background()
	mr, _, cache := newFenceFixture(t, &fenceStore{})

	first, ok := cache.generation(ctx, "u")
	require.True(t, ok)
	require.NotEmpty(t, first)
	again, _ := cache.generation(ctx, "u")
	require.Equal(t, first, again, "admission reuses the live token")

	require.NoError(t, cache.InvalidateByUser(ctx, "u"))
	require.True(t, mr.Exists(cache.holdKey("u")))
	require.LessOrEqual(t, mr.TTL(cache.holdKey("u")), cacheWriteHold)
	_, ok = cache.generation(ctx, "u")
	require.False(t, ok, "retrievals admitted during the hold are not cacheable")

	mr.FastForward(cacheWriteHold + time.Second)
	next, ok := cache.generation(ctx, "u")
	require.True(t, ok)
	require.NotEqual(t, first, next, "invalidation replaces the token")

	_, ok = cache.generation(ctx, "")
	require.False(t, ok, "results without an owner cannot be invalidated, so they are never cached")
}

func TestCachingStoreReadBegunDuringHoldIsNotCachedAfterIt(t *testing.T) {
	ctx := context.Background()
	read, gate := make(chan struct{}), make(chan struct{})
	store := &fenceStore{mem: &Memory{ID: "secret", UserID: "u"}, stale: true}
	mr, cached, cache := newFenceFixture(t, store)
	opts := RetrieveOptions{Query: "q", UserID: "u", Limit: 5}
	require.NoError(t, cached.Forget(ctx, "secret"))

	store.mu.Lock()
	store.read, store.gate = read, gate
	store.mu.Unlock()
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = cached.Retrieve(ctx, opts) // admitted during the hold, reads a stale row
	}()
	<-read
	mr.FastForward(cacheWriteHold + time.Second)
	close(gate)
	<-done

	_, hit := cache.Get(ctx, opts)
	require.False(t, hit, "a read begun during the hold must not be cached after it expires")
}

func TestCachingStoreReadSpanningFenceExpiryIsNotCached(t *testing.T) {
	ctx := context.Background()
	read, gate := make(chan struct{}), make(chan struct{})
	store := &fenceStore{mem: &Memory{ID: "secret", UserID: "u"}, stale: true, read: read, gate: gate}
	mr, cached, cache := newFenceFixture(t, store)
	opts := RetrieveOptions{Query: "q", UserID: "u", Limit: 5}

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = cached.Retrieve(ctx, opts) // admitted before Forget
	}()
	<-read
	require.NoError(t, cached.Forget(ctx, "secret"))
	mr.FastForward(2*cache.ttl + cacheWriteHold + time.Minute) // fence and hold keys expire
	close(gate)
	<-done

	_, hit := cache.Get(ctx, opts)
	require.False(t, hit, "an expired fence must not readmit a read that began before Forget")
}
