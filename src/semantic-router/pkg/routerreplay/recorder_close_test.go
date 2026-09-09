package routerreplay

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

type uncancelableOutcomeStore struct {
	store.Storage
	entered           chan struct{}
	release           chan struct{}
	closed            chan struct{}
	active            atomic.Bool
	closedWhileActive atomic.Bool
	closeCalls        atomic.Int32
}

func (s *uncancelableOutcomeStore) AppendOutcome(context.Context, string, store.Outcome) error {
	s.active.Store(true)
	defer s.active.Store(false)
	close(s.entered)
	<-s.release
	return nil
}

func (s *uncancelableOutcomeStore) Close() error {
	s.closedWhileActive.Store(s.active.Load())
	if s.closeCalls.Add(1) == 1 {
		close(s.closed)
	}
	return nil
}

func TestRecorderCloseRetainsStoresUntilSharedWorkerExits(t *testing.T) {
	blocked := &uncancelableOutcomeStore{
		entered: make(chan struct{}), release: make(chan struct{}), closed: make(chan struct{}),
	}
	other := &uncancelableOutcomeStore{closed: make(chan struct{})}
	first, second := NewRecorder(blocked), NewRecorder(other)
	ShareOutcomeQueue(first, second)
	first.outcomes.grace = 10 * time.Millisecond
	var once sync.Once
	unblock := func() { once.Do(func() { close(blocked.release) }) }
	t.Cleanup(func() { unblock(); _ = first.Close(); _ = second.Close() })
	require.True(t, first.TryAppendOutcome("record", Outcome{}))
	awaitOutcomeSignal(t, blocked.entered)
	// A shared queue owns events for both stores until its worker has exited.
	require.True(t, second.TryAppendOutcome("record", Outcome{}))
	closed := make(chan struct{})
	go func() {
		defer close(closed)
		assert.ErrorIs(t, first.Close(), context.DeadlineExceeded)
		assert.ErrorIs(t, second.Close(), context.DeadlineExceeded)
		assert.ErrorIs(t, first.Close(), context.DeadlineExceeded)
	}()
	awaitOutcomeSignal(t, closed)
	assert.True(t, blocked.active.Load())
	assert.Zero(t, blocked.closeCalls.Load())
	assert.Zero(t, other.closeCalls.Load())
	unblock()
	awaitOutcomeSignal(t, blocked.closed)
	awaitOutcomeSignal(t, other.closed)
	assert.False(t, blocked.closedWhileActive.Load())
	assert.EqualValues(t, 1, blocked.closeCalls.Load())
	assert.EqualValues(t, 1, other.closeCalls.Load())
	require.ErrorIs(t, first.Close(), context.DeadlineExceeded)
	assert.EqualValues(t, 1, blocked.closeCalls.Load())
}
