package extproc

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

func TestResourceScopeClosesAfterAbandonedPersistenceReservation(t *testing.T) {
	runner := memory.NewPersistenceRunner(time.Minute, 1, 1)
	reservation := runner.TryReserve(context.Background(), func(string, string, bool, error) {})
	require.NotNil(t, reservation)
	t.Cleanup(func() {
		reservation.Abort(memory.PersistenceOutcome{}, nil)
		_ = runner.RetireAndWait(time.Millisecond)
	})
	scope := newResourceScope()
	closed := make(chan struct{})
	scope.add(func() error { close(closed); return nil })
	scope.addDraining(func() error { return runner.RetireAndWait(time.Millisecond) }, runner.Done())
	require.ErrorIs(t, scope.close(), memory.ErrPersistenceShutdownDeadline)
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("abandoned reservation leaked generation resources")
	}
	reservation.Start(func(context.Context) (memory.PersistenceOutcome, error) {
		t.Error("late work ran after its resources closed")
		return memory.PersistenceOutcome{}, nil
	})
	require.NoError(t, scope.close())
}

func TestResourceScopeRetainsGenerationUntilPersistenceExits(t *testing.T) {
	runner := memory.NewPersistenceRunner(time.Minute, 1, 1)
	scope := newResourceScope()
	entered, release, storeClosed := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	var active, closedWhileActive atomic.Bool
	var closeCalls atomic.Int32
	scope.add(func() error {
		closedWhileActive.Store(active.Load())
		closeCalls.Add(1)
		close(storeClosed)
		return nil
	})
	scope.addDraining(func() error {
		return runner.RetireAndWait(time.Millisecond)
	}, runner.Done())
	t.Cleanup(func() { unblock(); _ = scope.close() })
	runner.Submit(context.Background(), memory.PersistenceJob{
		Run: func(context.Context) (memory.PersistenceOutcome, error) {
			active.Store(true)
			defer active.Store(false)
			close(entered)
			<-release
			return memory.PersistenceOutcome{}, nil
		},
		Report: func(string, string, bool, error) {},
	})
	<-entered
	require.ErrorIs(t, scope.close(), memory.ErrPersistenceShutdownDeadline)
	require.NoError(t, scope.close())
	assert.True(t, active.Load())
	assert.Zero(t, closeCalls.Load())
	unblock()
	select {
	case <-storeClosed:
	case <-time.After(time.Second):
		t.Fatal("deferred generation cleanup did not finish")
	}
	assert.False(t, closedWhileActive.Load())
	assert.EqualValues(t, 1, closeCalls.Load())
}
