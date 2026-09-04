package memory

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type outcomeRecorder struct {
	mu       sync.Mutex
	outcomes map[string]int
	failOpen map[string]bool
	events   []string
}

func newOutcomeRecorder() *outcomeRecorder {
	return &outcomeRecorder{outcomes: map[string]int{}, failOpen: map[string]bool{}}
}

func (o *outcomeRecorder) report(status, reason string, failOpen bool, _ error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	key := status + "/" + reason
	o.outcomes[key]++
	o.failOpen[key] = failOpen
	o.events = append(o.events, key)
}

func (o *outcomeRecorder) count(key string) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.outcomes[key]
}

func (o *outcomeRecorder) snapshot() map[string]int {
	o.mu.Lock()
	defer o.mu.Unlock()
	copied := make(map[string]int, len(o.outcomes))
	for k, v := range o.outcomes {
		copied[k] = v
	}
	return copied
}

func (o *outcomeRecorder) failedOpen(key string) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.failOpen[key]
}

func (o *outcomeRecorder) eventSnapshot() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]string(nil), o.events...)
}

func succeedingJob(o *outcomeRecorder) PersistenceJob {
	return PersistenceJob{
		Run:    func(context.Context) (PersistenceOutcome, error) { return PersistenceOutcome{}, nil },
		Report: o.report,
	}
}

func blockingJob(o *outcomeRecorder) PersistenceJob {
	return PersistenceJob{
		Run: func(ctx context.Context) (PersistenceOutcome, error) {
			<-ctx.Done()
			return PersistenceOutcome{}, ctx.Err()
		},
		Report: o.report,
	}
}

func TestPersistenceRunner_Success(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 2, 8)

	runner.Submit(context.Background(), succeedingJob(recorder))
	require.NoError(t, runner.RetireAndWait(time.Second))

	assert.Equal(t, 1, recorder.count("completed/persisted"), recorder.snapshot())
	assert.False(t, recorder.failedOpen("completed/persisted"))
	assert.Equal(t, []string{"scheduled/queue_accepted", "completed/persisted"}, recorder.eventSnapshot())
}

func TestPersistenceRunner_WaitsForScheduledReceiptBeforeRunning(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 1, 1)
	scheduledStarted := make(chan struct{})
	releaseScheduled := make(chan struct{})
	runStarted := make(chan struct{})

	runner.Submit(context.Background(), PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) {
			close(runStarted)
			return PersistenceOutcome{}, nil
		},
		Report: func(status, reason string, failOpen bool, cause error) {
			if status == "scheduled" {
				close(scheduledStarted)
				<-releaseScheduled
			}
			recorder.report(status, reason, failOpen, cause)
		},
	})

	<-scheduledStarted
	select {
	case <-runStarted:
		t.Fatal("worker ran before scheduled receipt completed")
	case <-time.After(25 * time.Millisecond):
	}

	close(releaseScheduled)
	require.NoError(t, runner.RetireAndWait(time.Second))
	assert.Equal(t, []string{"scheduled/queue_accepted", "completed/persisted"}, recorder.eventSnapshot())
}

func TestPersistenceRunner_BackendFailure(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 2, 8)

	runner.Submit(context.Background(), PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) {
			return PersistenceOutcome{}, errors.New("backend unavailable")
		},
		Report: recorder.report,
	})
	require.NoError(t, runner.RetireAndWait(time.Second))

	assert.Equal(t, 1, recorder.count("store_failed/persist_error"), recorder.snapshot())
	assert.True(t, recorder.failedOpen("store_failed/persist_error"),
		"a swallowed backend failure must be marked fail-open so alerting can find it")
}

func TestPersistenceRunner_PreservesJobOutcome(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 1, 1)
	cause := errors.New("failed to encode history")
	var reportedCause error

	runner.Submit(context.Background(), PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) {
			return PersistenceOutcome{
				Status:   "extraction_failed",
				Reason:   "history_encode_error",
				FailOpen: true,
			}, cause
		},
		Report: func(status, reason string, failOpen bool, gotCause error) {
			recorder.report(status, reason, failOpen, gotCause)
			reportedCause = gotCause
		},
	})
	require.NoError(t, runner.RetireAndWait(time.Second))

	assert.Equal(t, 1, recorder.count("extraction_failed/history_encode_error"), recorder.snapshot())
	assert.True(t, recorder.failedOpen("extraction_failed/history_encode_error"))
	assert.ErrorIs(t, reportedCause, cause)
}

func TestPersistenceRunner_Timeout(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(30*time.Millisecond, 2, 8)

	runner.Submit(context.Background(), blockingJob(recorder))
	require.Eventually(t, func() bool {
		return recorder.count("timeout/persist_timeout") == 1
	}, time.Second, 5*time.Millisecond, recorder.snapshot())

	require.NoError(t, runner.RetireAndWait(time.Second))
	assert.True(t, recorder.failedOpen("timeout/persist_timeout"))
}

func TestPersistenceRunner_SubmitterCancellationDoesNotAbortWrite(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 2, 8)

	traceCtx, cancelTrace := context.WithCancel(context.Background())
	started := make(chan struct{})
	runner.Submit(traceCtx, PersistenceJob{
		Run: func(ctx context.Context) (PersistenceOutcome, error) {
			close(started)
			time.Sleep(30 * time.Millisecond)
			return PersistenceOutcome{}, ctx.Err()
		},
		Report: recorder.report,
	})
	<-started
	cancelTrace()

	require.NoError(t, runner.RetireAndWait(time.Second))
	assert.Equal(t, 1, recorder.count("completed/persisted"), recorder.snapshot())
}

func TestPersistenceRunner_ShedsWhenQueueIsFull(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 1, 1)

	release := make(chan struct{})
	occupy := PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) {
			<-release
			return PersistenceOutcome{}, nil
		},
		Report: recorder.report,
	}

	runner.Submit(context.Background(), occupy)
	require.Eventually(t, func() bool {
		return len(runner.jobs) == 0
	}, time.Second, 5*time.Millisecond, "worker never picked up the first job")

	runner.Submit(context.Background(), occupy)
	runner.Submit(context.Background(), succeedingJob(recorder))

	assert.Equal(t, 1, recorder.count("rejected/queue_full"), recorder.snapshot())
	assert.False(t, recorder.failedOpen("rejected/queue_full"),
		"shedding is a capacity policy, not a swallowed failure")

	close(release)
	require.NoError(t, runner.RetireAndWait(2*time.Second))
	events := recorder.eventSnapshot()
	scheduled := 0
	for _, event := range events {
		if event == "scheduled/queue_accepted" {
			scheduled++
		}
	}
	assert.Equal(t, 2, scheduled, events)
}

func TestPersistenceRunner_HonorsConcurrencyLimit(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(5*time.Second, 2, 32)

	var mu sync.Mutex
	active, peak := 0, 0
	for i := 0; i < 12; i++ {
		runner.Submit(context.Background(), PersistenceJob{
			Run: func(context.Context) (PersistenceOutcome, error) {
				mu.Lock()
				active++
				if active > peak {
					peak = active
				}
				mu.Unlock()
				time.Sleep(10 * time.Millisecond)
				mu.Lock()
				active--
				mu.Unlock()
				return PersistenceOutcome{}, nil
			},
			Report: recorder.report,
		})
	}
	require.NoError(t, runner.RetireAndWait(5*time.Second))

	assert.Equal(t, 12, recorder.count("completed/persisted"), recorder.snapshot())
	mu.Lock()
	defer mu.Unlock()
	assert.LessOrEqual(t, peak, 2, "concurrency limit was exceeded")
}

func TestPersistenceRunner_ShutdownDrainsQueuedWork(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 1, 16)

	for i := 0; i < 5; i++ {
		runner.Submit(context.Background(), succeedingJob(recorder))
	}
	require.NoError(t, runner.RetireAndWait(2*time.Second))

	assert.Equal(t, 5, recorder.count("completed/persisted"), recorder.snapshot())
}

func TestPersistenceRunner_CancelsInFlightWorkOnceGraceElapses(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(10*time.Second, 2, 8)

	runner.Submit(context.Background(), blockingJob(recorder))
	require.Eventually(t, func() bool {
		return len(runner.jobs) == 0
	}, time.Second, 5*time.Millisecond, "worker never picked up the job")

	err := runner.RetireAndWait(50 * time.Millisecond)
	require.ErrorIs(t, err, ErrPersistenceShutdownDeadline)
	assert.Equal(t, 1, recorder.count("cancelled/shutdown"), recorder.snapshot())
}

func TestPersistenceRunner_RejectsSubmitAfterRetire(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 2, 8)
	require.NoError(t, runner.RetireAndWait(time.Second))

	runner.Submit(context.Background(), succeedingJob(recorder))

	assert.Equal(t, 1, recorder.count("rejected/shutting_down"), recorder.snapshot())
	assert.Equal(t, []string{"rejected/shutting_down"}, recorder.eventSnapshot())
}

func TestPersistenceRunner_RetireIsIdempotent(t *testing.T) {
	runner := NewPersistenceRunner(time.Second, 2, 8)
	require.NoError(t, runner.RetireAndWait(time.Second))
	require.NoError(t, runner.RetireAndWait(time.Second))
}

func TestPersistenceRunner_ContainsPanic(t *testing.T) {
	recorder := newOutcomeRecorder()
	runner := NewPersistenceRunner(time.Second, 2, 8)

	runner.Submit(context.Background(), PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) {
			panic("payload of an unexpected shape")
		},
		Report: recorder.report,
	})
	require.NoError(t, runner.RetireAndWait(time.Second))

	assert.Equal(t, 1, recorder.count("store_failed/panic"), recorder.snapshot())
}

func TestPersistenceRunner_ContainsReportPanic(t *testing.T) {
	runner := NewPersistenceRunner(time.Second, 1, 4)
	var mu sync.Mutex
	var seen []string

	runner.Submit(context.Background(), PersistenceJob{
		Run: func(context.Context) (PersistenceOutcome, error) { return PersistenceOutcome{}, nil },
		Report: func(status, _ string, _ bool, _ error) {
			mu.Lock()
			seen = append(seen, status)
			mu.Unlock()
			panic("replay recorder exploded")
		},
	})
	require.NoError(t, runner.RetireAndWait(time.Second))

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, []string{"scheduled", "completed"}, seen)
}

func TestPersistenceRunner_ContainsRejectionReportPanic(t *testing.T) {
	runner := NewPersistenceRunner(time.Second, 1, 1)
	require.NoError(t, runner.RetireAndWait(time.Second))

	assert.NotPanics(t, func() {
		runner.Submit(context.Background(), PersistenceJob{
			Run:    func(context.Context) (PersistenceOutcome, error) { return PersistenceOutcome{}, nil },
			Report: func(string, string, bool, error) { panic("metrics registry exploded") },
		})
	})
}

func TestPersistenceRunner_PanicHandlerSurvivesPanickingReport(t *testing.T) {
	runner := NewPersistenceRunner(time.Second, 1, 1)
	reports := make(chan string, 4)

	assert.NotPanics(t, func() {
		runner.Submit(context.Background(), PersistenceJob{
			Run: func(context.Context) (PersistenceOutcome, error) { panic("payload of an unexpected shape") },
			Report: func(status, _ string, _ bool, _ error) {
				reports <- status
				panic("reporting exploded too")
			},
		})
		require.NoError(t, runner.RetireAndWait(time.Second))
	})

	close(reports)
	var seen []string
	for status := range reports {
		seen = append(seen, status)
	}
	assert.Equal(t, []string{"scheduled", "store_failed"}, seen)
}

func TestPersistenceRunner_NonPositiveBoundsFallBackToDefaults(t *testing.T) {
	runner := NewPersistenceRunner(0, 0, 0)
	defer func() { _ = runner.RetireAndWait(time.Second) }()

	assert.Equal(t, DefaultPersistenceTimeout, runner.timeout)
	assert.Equal(t, DefaultPersistenceQueue, cap(runner.jobs))
}
