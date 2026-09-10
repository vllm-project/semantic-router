package memory

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPersistenceRunnerTimeoutRetainsUncooperativeWorker(t *testing.T) {
	for _, panics := range []bool{false, true} {
		t.Run(map[bool]string{false: "late_success", true: "late_panic"}[panics], func(t *testing.T) {
			runner := NewPersistenceRunner(20*time.Millisecond, 1, 1)
			receipts := newOutcomeRecorder()
			entered, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			t.Cleanup(func() { unblock(); _ = runner.RetireAndWait(time.Second) })
			runner.Submit(context.Background(), PersistenceJob{
				Run: func(context.Context) (PersistenceOutcome, error) {
					close(entered)
					<-release // Deliberately ignores cancellation, like a native call.
					if panics {
						panic("late native failure")
					}
					return PersistenceOutcome{}, nil
				},
				Report: receipts.report,
			})
			<-entered
			require.Eventually(t, func() bool {
				return receipts.count("timeout/persist_timeout") == 1
			}, time.Second, time.Millisecond)
			var queuedRuns atomic.Int32
			queued := PersistenceJob{
				Run: func(context.Context) (PersistenceOutcome, error) {
					queuedRuns.Add(1)
					return PersistenceOutcome{}, nil
				},
				Report: receipts.report,
			}
			runner.Submit(context.Background(), queued)
			runner.Submit(context.Background(), queued)
			assert.Equal(t, 1, receipts.count("rejected/queue_full"))
			assert.Zero(t, queuedRuns.Load(), "timeout must not free the occupied slot")
			unblock()
			require.NoError(t, runner.RetireAndWait(time.Second))
			assert.Equal(t, 1, receipts.count("timeout/persist_timeout"))
			assert.Equal(t, 1, receipts.count("completed/persisted"), "only the queued job completes")
			assert.Zero(t, receipts.count("store_failed/panic"), "no second terminal receipt")
		})
	}
}

func TestPersistenceRunnerShutdownRetainsUncooperativeWorker(t *testing.T) {
	runner := NewPersistenceRunner(time.Minute, 1, 1)
	receipts := newOutcomeRecorder()
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(func() { unblock(); _ = runner.RetireAndWait(time.Second); <-runner.Done() })
	runner.Submit(context.Background(), PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) {
			close(entered)
			<-release
			return PersistenceOutcome{}, nil
		},
		Report: receipts.report,
	})
	<-entered
	var queuedRuns atomic.Int32
	runner.Submit(context.Background(), PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) {
			queuedRuns.Add(1)
			return PersistenceOutcome{}, nil
		},
		Report: receipts.report,
	})
	require.ErrorIs(t, runner.RetireAndWait(time.Millisecond), ErrPersistenceShutdownDeadline)
	require.ErrorIs(t, runner.RetireAndWait(time.Millisecond), ErrPersistenceShutdownDeadline)
	select {
	case <-runner.Done():
		t.Fatal("retirement claimed completion while Run is still active")
	default:
	}
	require.Eventually(t, func() bool {
		return receipts.count("cancelled/shutdown") == 2
	}, time.Second, time.Millisecond, "queued work must report cancellation before the active job exits")
	unblock()
	select {
	case <-runner.Done():
	case <-time.After(time.Second):
		t.Fatal("workers did not exit after release")
	}
	assert.Zero(t, queuedRuns.Load(), "canceled queued jobs must not start")
	assert.Equal(t, 2, receipts.count("cancelled/shutdown"))
	assert.Zero(t, receipts.count("completed/persisted"))
}

func TestPersistenceRunnerQueuedTimeoutDoesNotWaitForWorker(t *testing.T) {
	runner := NewPersistenceRunner(20*time.Millisecond, 1, 1)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	t.Cleanup(func() { unblock(); _ = runner.RetireAndWait(time.Second); <-runner.Done() })
	runner.Submit(context.Background(), PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) {
			close(entered)
			<-release
			return PersistenceOutcome{}, nil
		},
		Report: func(string, string, bool, error) {},
	})
	<-entered
	receipts := newOutcomeRecorder()
	var queuedRuns atomic.Int32
	runner.Submit(context.Background(), PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) {
			queuedRuns.Add(1)
			return PersistenceOutcome{}, nil
		},
		Report: receipts.report,
	})
	require.Eventually(t, func() bool {
		return receipts.count("timeout/persist_timeout") == 1
	}, time.Second, time.Millisecond, "a queued attempt must expire while the worker remains blocked")
	assert.True(t, receipts.failedOpen("timeout/persist_timeout"))
	assert.Zero(t, queuedRuns.Load())
	unblock()
	require.NoError(t, runner.RetireAndWait(time.Second))
	assert.Zero(t, queuedRuns.Load(), "expired queued work must never execute")
	assert.Equal(t, []string{"scheduled/queue_accepted", "timeout/persist_timeout"}, receipts.eventSnapshot())
}
