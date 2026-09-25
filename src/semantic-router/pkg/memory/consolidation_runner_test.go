package memory

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

type scriptMemoryStore struct {
	mu       sync.Mutex
	memories []*Memory
	listGate chan struct{}
	listErr  error
	stores   int
	forgets  int
	enabled  bool
}

func newScriptMemoryStore(memories ...*Memory) *scriptMemoryStore {
	copied := make([]*Memory, len(memories))
	copy(copied, memories)
	return &scriptMemoryStore{memories: copied, enabled: true}
}

func (s *scriptMemoryStore) Store(_ context.Context, memory *Memory) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stores++
	s.memories = append(s.memories, memory)
	return nil
}

func (s *scriptMemoryStore) Retrieve(context.Context, RetrieveOptions) ([]*RetrieveResult, error) {
	return nil, nil
}

func (s *scriptMemoryStore) Get(context.Context, string) (*Memory, error) { return nil, nil }

func (s *scriptMemoryStore) Update(context.Context, string, *Memory) error { return nil }

func (s *scriptMemoryStore) List(ctx context.Context, opts ListOptions) (*ListResult, error) {
	if s.listGate != nil {
		select {
		case <-s.listGate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if s.listErr != nil {
		return nil, s.listErr
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var matched []*Memory
	for _, memory := range s.memories {
		if memory.UserID == opts.UserID {
			matched = append(matched, memory)
		}
	}
	return &ListResult{Memories: matched, Total: len(matched), Limit: opts.Limit}, nil
}

func (s *scriptMemoryStore) Forget(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.forgets++
	kept := s.memories[:0]
	for _, memory := range s.memories {
		if memory.ID != id {
			kept = append(kept, memory)
		}
	}
	s.memories = kept
	return nil
}

func (s *scriptMemoryStore) ForgetByScope(context.Context, MemoryScope) error { return nil }
func (s *scriptMemoryStore) IsEnabled() bool                                  { return s.enabled }
func (s *scriptMemoryStore) CheckConnection(context.Context) error            { return nil }
func (s *scriptMemoryStore) Close() error                                     { return nil }

func consolidationCount(status, reason string) float64 {
	return testutil.ToFloat64(MemoryConsolidationTotal.WithLabelValues(status, reason))
}

func waitConsolidation(t *testing.T, status, reason string, before float64) {
	t.Helper()
	require.Eventually(t, func() bool {
		return consolidationCount(status, reason) > before
	}, 2*time.Second, 10*time.Millisecond)
}

func TestConsolidationRunnerMergesSimilarMemories(t *testing.T) {
	store := newScriptMemoryStore(
		&Memory{ID: "a", UserID: "user-1", Content: "alpha beta gamma", Type: MemoryTypeSemantic},
		&Memory{ID: "b", UserID: "user-1", Content: "alpha beta gamma delta", Type: MemoryTypeSemantic},
	)
	before := consolidationCount("completed", "finished")
	runner := NewConsolidationRunner(store, ConsolidationOptions{Cooldown: time.Hour, Timeout: time.Second, Concurrency: 1})
	t.Cleanup(func() { _ = runner.RetireAndWait(time.Second) })

	runner.Enqueue("user-1")
	waitConsolidation(t, "completed", "finished", before)
	require.Equal(t, 1, store.stores)
	require.Equal(t, 2, store.forgets)
}

func TestConsolidationRunnerNoOpWithOneMemory(t *testing.T) {
	store := newScriptMemoryStore(&Memory{ID: "a", UserID: "user-1", Content: "only one"})
	beforeMerged := testutil.ToFloat64(MemoryConsolidationMerged)
	before := consolidationCount("completed", "finished")
	runner := NewConsolidationRunner(store, ConsolidationOptions{Cooldown: time.Hour, Timeout: time.Second, Concurrency: 1})
	t.Cleanup(func() { _ = runner.RetireAndWait(time.Second) })

	runner.Enqueue("user-1")
	waitConsolidation(t, "completed", "finished", before)
	require.Equal(t, beforeMerged, testutil.ToFloat64(MemoryConsolidationMerged))
	require.Equal(t, 0, store.stores)
	require.Equal(t, 0, store.forgets)
}

func TestConsolidationRunnerRecordsStoreFailure(t *testing.T) {
	store := newScriptMemoryStore()
	store.listErr = errors.New("backend down")
	before := consolidationCount("failed", "consolidate_error")
	runner := NewConsolidationRunner(store, ConsolidationOptions{Cooldown: time.Hour, Timeout: time.Second, Concurrency: 1})
	t.Cleanup(func() { _ = runner.RetireAndWait(time.Second) })

	runner.Enqueue("user-1")
	waitConsolidation(t, "failed", "consolidate_error", before)
}

func TestConsolidationRunnerCooldownSkipsSecondEnqueue(t *testing.T) {
	store := newScriptMemoryStore(&Memory{ID: "a", UserID: "user-1", Content: "only one"})
	completed := consolidationCount("completed", "finished")
	skipped := consolidationCount("skipped", "cooldown")
	runner := NewConsolidationRunner(store, ConsolidationOptions{Cooldown: time.Hour, Timeout: time.Second, Concurrency: 1})
	t.Cleanup(func() { _ = runner.RetireAndWait(time.Second) })

	runner.Enqueue("user-1")
	waitConsolidation(t, "completed", "finished", completed)
	runner.Enqueue("user-1")
	waitConsolidation(t, "skipped", "cooldown", skipped)
}

func TestConsolidationRunnerRejectsWhenBusy(t *testing.T) {
	store := newScriptMemoryStore(&Memory{ID: "a", UserID: "user-1", Content: "one"})
	store.listGate = make(chan struct{})
	rejected := consolidationCount("rejected", "concurrency_full")
	runner := NewConsolidationRunner(store, ConsolidationOptions{Cooldown: time.Millisecond, Timeout: time.Second, Concurrency: 1})
	t.Cleanup(func() {
		close(store.listGate)
		_ = runner.RetireAndWait(time.Second)
	})

	runner.Enqueue("user-1")
	require.Eventually(t, func() bool {
		runner.mu.Lock()
		defer runner.mu.Unlock()
		_, ok := runner.inflight["user-1"]
		return ok
	}, time.Second, 5*time.Millisecond)
	runner.Enqueue("user-2")
	waitConsolidation(t, "rejected", "concurrency_full", rejected)
}

func TestConsolidationRunnerTimeoutCancelsList(t *testing.T) {
	store := newScriptMemoryStore()
	store.listGate = make(chan struct{})
	before := consolidationCount("timeout", "consolidate_timeout")
	runner := NewConsolidationRunner(store, ConsolidationOptions{Cooldown: time.Hour, Timeout: 20 * time.Millisecond, Concurrency: 1})
	t.Cleanup(func() {
		select {
		case <-store.listGate:
		default:
			close(store.listGate)
		}
		_ = runner.RetireAndWait(time.Second)
	})

	runner.Enqueue("user-1")
	waitConsolidation(t, "timeout", "consolidate_timeout", before)
}

func TestConsolidationRunnerShutdownDoesNotHang(t *testing.T) {
	store := newScriptMemoryStore()
	store.listGate = make(chan struct{})
	runner := NewConsolidationRunner(store, ConsolidationOptions{Cooldown: time.Hour, Timeout: time.Minute, Concurrency: 1})
	runner.Enqueue("user-1")
	require.Eventually(t, func() bool {
		runner.mu.Lock()
		defer runner.mu.Unlock()
		_, ok := runner.inflight["user-1"]
		return ok
	}, time.Second, 5*time.Millisecond)

	finished := make(chan error, 1)
	go func() { finished <- runner.RetireAndWait(50 * time.Millisecond) }()
	select {
	case err := <-finished:
		require.NoError(t, err)
	case <-time.After(2 * time.Second):
		t.Fatal("RetireAndWait hung")
	}
	close(store.listGate)
	select {
	case <-runner.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("runner did not finish after cancel")
	}
}

func TestConsolidationEnqueueNilAndEmptyAreNoops(t *testing.T) {
	var runner *ConsolidationRunner
	runner.Enqueue("user-1")
	live := NewConsolidationRunner(newScriptMemoryStore(), ConsolidationOptions{Concurrency: 1})
	t.Cleanup(func() { _ = live.RetireAndWait(time.Second) })
	live.Enqueue("")
}

func TestConsolidationRunnerEvictsExpiredCooldownEntries(t *testing.T) {
	store := newScriptMemoryStore(&Memory{ID: "a", UserID: "user-1", Content: "only one"})
	runner := NewConsolidationRunner(store, ConsolidationOptions{
		Cooldown:    20 * time.Millisecond,
		Timeout:     time.Second,
		Concurrency: 1,
	})
	t.Cleanup(func() { _ = runner.RetireAndWait(time.Second) })

	before := consolidationCount("completed", "finished")
	runner.Enqueue("user-1")
	waitConsolidation(t, "completed", "finished", before)

	runner.mu.Lock()
	require.Contains(t, runner.lastAccepted, "user-1")
	runner.mu.Unlock()

	require.Eventually(t, func() bool {
		runner.Enqueue("user-2")
		runner.mu.Lock()
		defer runner.mu.Unlock()
		_, kept := runner.lastAccepted["user-1"]
		return !kept
	}, time.Second, 5*time.Millisecond)
}
