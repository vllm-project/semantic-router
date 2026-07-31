package extproc

import (
	"context"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// closeTrackingCache is a minimal CacheBackend stub that records how many
// times Close() was invoked.
type closeTrackingCache struct {
	mu         sync.Mutex
	closed     bool
	closeCalls int
}

func (c *closeTrackingCache) IsEnabled() bool                                         { return true }
func (c *closeTrackingCache) CheckConnection() error                                  { return nil }
func (c *closeTrackingCache) LastSimilarity() float32                                 { return 0 }
func (c *closeTrackingCache) GetStats() cache.CacheStats                              { return cache.CacheStats{} }
func (c *closeTrackingCache) AddPendingRequest(_, _, _ string, _ []byte, _ int) error { return nil }
func (c *closeTrackingCache) UpdateWithResponse(_ string, _ []byte, _ int) error      { return nil }
func (c *closeTrackingCache) AddEntry(_, _, _ string, _, _ []byte, _ int) error       { return nil }
func (c *closeTrackingCache) FindSimilar(_, _ string) ([]byte, bool, error)           { return nil, false, nil }

func (c *closeTrackingCache) FindSimilarWithThreshold(_, _ string, _ float32) ([]byte, bool, error) {
	return nil, false, nil
}

func (c *closeTrackingCache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.closed = true
	c.closeCalls++
	return nil
}

func (c *closeTrackingCache) closeCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closeCalls
}

// closeTrackingMemoryStore is a minimal memory.Store stub that records
// whether Close() was ever invoked.
type closeTrackingMemoryStore struct {
	closed bool
}

func (m *closeTrackingMemoryStore) Store(context.Context, *memory.Memory) error { return nil }
func (m *closeTrackingMemoryStore) Retrieve(context.Context, memory.RetrieveOptions) ([]*memory.RetrieveResult, error) {
	return nil, nil
}

func (m *closeTrackingMemoryStore) Get(context.Context, string) (*memory.Memory, error) {
	return nil, nil
}

func (m *closeTrackingMemoryStore) Update(context.Context, string, *memory.Memory) error { return nil }

func (m *closeTrackingMemoryStore) List(context.Context, memory.ListOptions) (*memory.ListResult, error) {
	return nil, nil
}
func (m *closeTrackingMemoryStore) Forget(context.Context, string) error { return nil }
func (m *closeTrackingMemoryStore) ForgetByScope(context.Context, memory.MemoryScope) error {
	return nil
}
func (m *closeTrackingMemoryStore) IsEnabled() bool                       { return true }
func (m *closeTrackingMemoryStore) CheckConnection(context.Context) error { return nil }

func (m *closeTrackingMemoryStore) Close() error {
	m.closed = true
	return nil
}

// closeTrackingReplayStorage is a minimal replay store.Storage stub that
// records whether Close() was ever invoked.
type closeTrackingReplayStorage struct {
	closed bool
}

func (s *closeTrackingReplayStorage) Add(context.Context, store.Record) (string, error) {
	return "", nil
}

func (s *closeTrackingReplayStorage) UpdateStatus(context.Context, string, int, bool, bool) error {
	return nil
}

func (s *closeTrackingReplayStorage) AttachRequest(context.Context, string, string, bool) error {
	return nil
}

func (s *closeTrackingReplayStorage) AttachResponse(context.Context, string, string, bool) error {
	return nil
}

func (s *closeTrackingReplayStorage) AppendOutcome(context.Context, string, store.Outcome) error {
	return nil
}

func (s *closeTrackingReplayStorage) Get(context.Context, string) (store.Record, bool, error) {
	return store.Record{}, false, nil
}

func (s *closeTrackingReplayStorage) List(context.Context) ([]store.Record, error) { return nil, nil }

func (s *closeTrackingReplayStorage) UpdateHallucinationStatus(context.Context, string, bool, float32, []string, []store.HallucinationSpan) error {
	return nil
}

func (s *closeTrackingReplayStorage) UpdateUsageCost(context.Context, string, store.UsageCost) error {
	return nil
}

func (s *closeTrackingReplayStorage) UpdateToolTrace(context.Context, string, store.ToolTrace) error {
	return nil
}

func (s *closeTrackingReplayStorage) Close() error {
	s.closed = true
	return nil
}

// TestOpenAIRouterCloseClosesOwnedResources asserts OpenAIRouter.Close()
// closes every owned resource, not just lookupTableCancel. This router is
// assembled by hand rather than by buildRouterComponents, so it exercises
// Close's field-by-field fallback path (no generation to delegate to).
func TestOpenAIRouterCloseClosesOwnedResources(t *testing.T) {
	fakeCache := &closeTrackingCache{}
	fakeMemStore := &closeTrackingMemoryStore{}
	fakeStorage := &closeTrackingReplayStorage{}
	recorder := routerreplay.NewRecorder(fakeStorage)

	router := &OpenAIRouter{
		Cache:          fakeCache,
		MemoryStore:    fakeMemStore,
		ReplayRecorder: recorder,
	}

	if err := router.Close(); err != nil {
		t.Fatalf("router.Close() error = %v", err)
	}

	if !fakeCache.closed {
		t.Error("OpenAIRouter.Close() did not close Cache")
	}
	if !fakeMemStore.closed {
		t.Error("OpenAIRouter.Close() did not close MemoryStore")
	}
	if !fakeStorage.closed {
		t.Error("OpenAIRouter.Close() did not close ReplayRecorder")
	}
}

// TestOpenAIRouterCloseIsIdempotent asserts a router closes its resources
// exactly once no matter how many times, or from how many goroutines, Close
// is called. A router is reachable from more than one shutdown path — a
// reload retiring the lease it replaced, and process shutdown retiring
// whatever lease is current — and not every underlying resource tolerates a
// second Close (a gRPC ClientConn, an MCP client).
func TestOpenAIRouterCloseIsIdempotent(t *testing.T) {
	fakeCache := &closeTrackingCache{}
	router := &OpenAIRouter{Cache: fakeCache}

	const closers = 8
	var wg sync.WaitGroup
	wg.Add(closers)
	errs := make([]error, closers)
	for i := 0; i < closers; i++ {
		go func() {
			defer wg.Done()
			errs[i] = router.Close()
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent Close() #%d error = %v", i, err)
		}
	}
	if got := fakeCache.closeCount(); got != 1 {
		t.Fatalf("Cache.Close() called %d times, want exactly 1", got)
	}
}

// TestOpenAIRouterCloseDelegatesToOwningGeneration asserts a router built by
// buildRouterComponents tears down through its generation instead of the
// field-by-field fallback. That is what keeps a single list of what a router
// owns: if Close also walked the fields, every new resource would have to be
// registered in two places and could silently drift out of sync.
func TestOpenAIRouterCloseDelegatesToOwningGeneration(t *testing.T) {
	generationClosers := 0
	gen := routerruntime.NewGeneration()
	gen.Defer(func() error {
		generationClosers++
		return nil
	})

	fakeCache := &closeTrackingCache{}
	router := &OpenAIRouter{Cache: fakeCache, generation: gen}

	if err := router.Close(); err != nil {
		t.Fatalf("router.Close() error = %v", err)
	}

	if generationClosers != 1 {
		t.Fatalf("generation closers ran %d times, want exactly 1", generationClosers)
	}
	if got := fakeCache.closeCount(); got != 0 {
		t.Fatalf("Close() bypassed the generation and closed Cache directly %d times", got)
	}
}
