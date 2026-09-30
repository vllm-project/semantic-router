package routerreplay

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

type gatedOutcomeStore struct {
	store.Storage
	entered chan struct{}
	release chan struct{}
	closed  bool
	mu      sync.Mutex
	seen    []Outcome
}

func (s *gatedOutcomeStore) AppendOutcome(ctx context.Context, _ string, outcome store.Outcome) error {
	select {
	case s.entered <- struct{}{}:
	default:
	}
	select {
	case <-s.release:
	case <-ctx.Done():
		return ctx.Err()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seen = append(s.seen, outcome)
	return nil
}

func (s *gatedOutcomeStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}

func awaitOutcomeSignal(t *testing.T, signal <-chan struct{}) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(time.Second):
		t.Fatal("outcome worker did not signal")
	}
}

func TestRecorderAsyncOutcomeSharedCapacitySnapshotAndClose(t *testing.T) {
	backend := &gatedOutcomeStore{entered: make(chan struct{}, 1), release: make(chan struct{})}
	first, second := NewRecorder(backend), NewRecorder(backend)
	assert.Equal(t, outcomeShutdownGrace, first.outcomes.grace)
	ShareOutcomeQueue(first, second)
	assert.Equal(t, outcomeShutdownGrace, first.outcomes.grace)
	require.Same(t, first.outcomes, second.outcomes)
	assert.False(t, first.outcomes.started, "writer should be lazy")
	var once sync.Once
	release := func() { once.Do(func() { close(backend.release) }) }
	t.Cleanup(func() { release(); assert.NoError(t, first.DrainOutcomes()) })
	require.True(t, first.TryAppendOutcome("record", Outcome{Verdict: "scheduled"}))
	awaitOutcomeSignal(t, backend.entered)
	metadata := map[string]string{"original": "value"}
	require.True(t, second.TryAppendOutcome("record", Outcome{Verdict: "terminal", Metadata: metadata}))
	metadata["original"] = "changed"
	for i := 1; i < DefaultOutcomeQueueCapacity; i++ {
		require.True(t, first.TryAppendOutcome("record", Outcome{}))
	}
	require.False(t, second.TryAppendOutcome("record", Outcome{}), "capacity is shared across recorders")
	release()
	require.NoError(t, first.Close())
	require.False(t, second.TryAppendOutcome("record", Outcome{}))
	backend.mu.Lock()
	defer backend.mu.Unlock()
	assert.True(t, backend.closed)
	require.Len(t, backend.seen, DefaultOutcomeQueueCapacity+1)
	assert.Equal(t, "scheduled", backend.seen[0].Verdict)
	assert.Equal(t, "terminal", backend.seen[1].Verdict)
	assert.Equal(t, "value", backend.seen[1].Metadata["original"])
}

func TestRecorderAsyncOutcomeShutdownCancelsStalledWriter(t *testing.T) {
	backend := &gatedOutcomeStore{entered: make(chan struct{}, 1), release: make(chan struct{})}
	recorder := NewRecorder(backend)
	ShareOutcomeQueue(recorder)
	// Shorten only this test's internal budget to exercise cancellation quickly.
	recorder.outcomes.grace = 10 * time.Millisecond
	t.Cleanup(func() { close(backend.release); _ = recorder.DrainOutcomes() })
	require.True(t, recorder.TryAppendOutcome("record", Outcome{}))
	awaitOutcomeSignal(t, backend.entered)
	require.True(t, recorder.TryAppendOutcome("record", Outcome{}))
	require.ErrorIs(t, recorder.DrainOutcomes(), context.DeadlineExceeded)
	awaitOutcomeSignal(t, recorder.outcomes.done)
	require.False(t, recorder.TryAppendOutcome("record", Outcome{}))
	require.ErrorIs(t, recorder.DrainOutcomes(), context.DeadlineExceeded)
	backend.mu.Lock()
	defer backend.mu.Unlock()
	assert.Empty(t, backend.seen)
}
