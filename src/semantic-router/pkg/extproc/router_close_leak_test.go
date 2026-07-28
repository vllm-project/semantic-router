package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// closeTrackingCache is a minimal CacheBackend stub that records whether
// Close() was ever invoked.
type closeTrackingCache struct {
	closed bool
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
	c.closed = true
	return nil
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

// TestOpenAIRouterCloseClosesOwnedResources documents that OpenAIRouter.Close()
// only cancels lookupTableCancel and never closes Cache, MemoryStore, or
// ReplayRecorder, even though all three already implement Close() error.
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
