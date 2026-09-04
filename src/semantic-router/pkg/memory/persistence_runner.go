package memory

import (
	"context"
	"errors"
	"runtime/debug"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	DefaultPersistenceTimeout       = 30 * time.Second
	DefaultPersistenceConcurrency   = 8
	DefaultPersistenceQueue         = 64
	DefaultPersistenceShutdownGrace = 5 * time.Second
	persistenceCancelUnwind         = time.Second
)

var ErrPersistenceShutdownDeadline = errors.New("memory persistence shutdown deadline exceeded")

type PersistenceOutcome struct {
	Status   string
	Reason   string
	FailOpen bool
}

type PersistenceJob struct {
	Run    func(ctx context.Context) (PersistenceOutcome, error)
	Report func(status, reason string, failOpen bool, cause error)
}

type queuedJob struct {
	traceCtx      context.Context
	job           PersistenceJob
	scheduledDone chan struct{}
}

type PersistenceRunner struct {
	jobs    chan queuedJob
	workers sync.WaitGroup
	timeout time.Duration

	baseCtx context.Context
	cancel  context.CancelFunc

	mu      sync.Mutex
	retired bool
}

func NewPersistenceRunner(timeout time.Duration, concurrency, queue int) *PersistenceRunner {
	if timeout <= 0 {
		timeout = DefaultPersistenceTimeout
	}
	if concurrency < 1 {
		concurrency = DefaultPersistenceConcurrency
	}
	if queue <= 0 {
		queue = DefaultPersistenceQueue
	}

	baseCtx, cancel := context.WithCancel(context.Background())
	r := &PersistenceRunner{
		jobs:    make(chan queuedJob, queue),
		timeout: timeout,
		baseCtx: baseCtx,
		cancel:  cancel,
	}

	r.workers.Add(concurrency)
	for i := 0; i < concurrency; i++ {
		go r.worker()
	}

	logging.ComponentEvent("memory", "persistence_runner_started", map[string]interface{}{
		"timeout_seconds": timeout.Seconds(),
		"concurrency":     concurrency,
		"queue":           queue,
	})
	return r
}

func reportSafely(job PersistenceJob, status, reason string, failOpen bool, cause error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			logging.ComponentErrorEvent("memory", "persistence_report_panic", map[string]interface{}{
				"status": status,
				"reason": reason,
				"panic":  recovered,
				"stack":  string(debug.Stack()),
			})
		}
	}()
	job.Report(status, reason, failOpen, cause)
}

func (r *PersistenceRunner) Submit(traceCtx context.Context, job PersistenceJob) {
	r.mu.Lock()
	if r.retired {
		r.mu.Unlock()
		reportSafely(job, "rejected", "shutting_down", false, nil)
		return
	}

	scheduledDone := make(chan struct{})
	select {
	case r.jobs <- queuedJob{traceCtx: traceCtx, job: job, scheduledDone: scheduledDone}:
		r.mu.Unlock()
		go func() {
			defer close(scheduledDone)
			reportSafely(job, "scheduled", "queue_accepted", false, nil)
		}()
	default:
		r.mu.Unlock()
		reportSafely(job, "rejected", "queue_full", false, nil)
	}
}

func (r *PersistenceRunner) worker() {
	defer r.workers.Done()
	for queued := range r.jobs {
		<-queued.scheduledDone
		r.run(queued.traceCtx, queued.job)
	}
}

func (r *PersistenceRunner) run(traceCtx context.Context, job PersistenceJob) {
	defer func() {
		if recovered := recover(); recovered != nil {
			logging.ComponentErrorEvent("memory", "persistence_panic", map[string]interface{}{
				"panic": recovered,
				"stack": string(debug.Stack()),
			})
			reportSafely(job, "store_failed", "panic", true, nil)
		}
	}()

	jobCtx, cancel := context.WithTimeout(context.WithoutCancel(traceCtx), r.timeout)
	defer cancel()
	stop := context.AfterFunc(r.baseCtx, cancel)
	defer stop()

	outcome, err := job.Run(jobCtx)

	switch {
	case errors.Is(jobCtx.Err(), context.DeadlineExceeded):
		reportSafely(job, "timeout", "persist_timeout", true, jobCtx.Err())
	case errors.Is(jobCtx.Err(), context.Canceled):
		reportSafely(job, "cancelled", "shutdown", true, jobCtx.Err())
	case outcome.Status != "":
		reportSafely(job, outcome.Status, outcome.Reason, outcome.FailOpen, err)
	case err != nil:
		reportSafely(job, "store_failed", "persist_error", true, err)
	default:
		reportSafely(job, "completed", "persisted", false, nil)
	}
}

func (r *PersistenceRunner) RetireAndWait(grace time.Duration) error {
	if grace <= 0 {
		grace = DefaultPersistenceShutdownGrace
	}

	r.mu.Lock()
	if r.retired {
		r.mu.Unlock()
		return nil
	}
	r.retired = true
	close(r.jobs)
	r.mu.Unlock()

	done := make(chan struct{})
	go func() {
		r.workers.Wait()
		close(done)
	}()

	select {
	case <-done:
		r.cancel()
		return nil
	case <-time.After(grace):
	}

	r.cancel()
	unwound := false
	select {
	case <-done:
		unwound = true
	case <-time.After(persistenceCancelUnwind):
	}
	logging.ComponentWarnEvent("memory", "persistence_shutdown_deadline", map[string]interface{}{
		"grace_seconds": grace.Seconds(),
		"unwound":       unwound,
	})
	return ErrPersistenceShutdownDeadline
}
