package memory

import (
	"context"
	"errors"
	"runtime/debug"
	"sync"
	"time"

	"go.opentelemetry.io/otel/trace"

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
	Run func(ctx context.Context) (PersistenceOutcome, error)
	// Report must only record local metrics or submit events without waiting for I/O.
	Report func(status, reason string, failOpen bool, cause error)
}

type queuedJob struct {
	ctx           context.Context
	job           PersistenceJob
	scheduledDone chan struct{}
	preparedDone  chan struct{}
	finish        func()
}

type PersistenceRunner struct {
	jobs    chan *queuedJob
	workers sync.WaitGroup
	timeout time.Duration

	baseCtx context.Context
	cancel  context.CancelFunc

	mu         sync.Mutex
	retired    bool
	retireOnce sync.Once
	retireErr  error
	done       chan struct{}
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
		jobs:    make(chan *queuedJob, queue),
		timeout: timeout,
		baseCtx: baseCtx,
		cancel:  cancel,
		done:    make(chan struct{}),
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
	if reservation := r.TryReserve(traceCtx, job.Report); reservation != nil {
		reservation.Start(job.Run)
	}
}

// TryReserve admits preparation before request-owned data is copied. A reserved
// job occupies queue/worker capacity until Start or Abort publishes its work.
// Callers must defer Abort so preparation failures cannot strand the reservation.
func (r *PersistenceRunner) TryReserve(traceCtx context.Context, report func(string, string, bool, error)) *PersistenceReservation {
	job := PersistenceJob{Report: report}
	r.mu.Lock()
	if r.retired {
		r.mu.Unlock()
		reportSafely(job, "rejected", "shutting_down", true, nil)
		return nil
	}

	queued := &queuedJob{job: job, scheduledDone: make(chan struct{}), preparedDone: make(chan struct{})}
	select {
	case r.jobs <- queued:
		// Start the deadline and cancellation reporter at acceptance, so queued
		// work reaches a terminal outcome even when every worker is stuck.
		r.prepareJob(traceCtx, queued)
		r.mu.Unlock()
		reportSafely(job, "scheduled", "queue_accepted", false, nil)
		close(queued.scheduledDone)
		return &PersistenceReservation{queued: queued}
	default:
		r.mu.Unlock()
		reportSafely(job, "rejected", "queue_full", true, nil)
		return nil
	}
}

type PersistenceReservation struct {
	queued *queuedJob
	once   sync.Once
}

func (r *PersistenceReservation) Context() context.Context { return r.queued.ctx }

func (r *PersistenceReservation) Start(run func(context.Context) (PersistenceOutcome, error)) {
	r.once.Do(func() {
		r.queued.job.Run = run
		close(r.queued.preparedDone)
	})
}

// Abort is harmless after Start. Until either is called, even a timed-out
// preparation retains capacity and generation resources.
func (r *PersistenceReservation) Abort(outcome PersistenceOutcome, err error) {
	r.Start(func(context.Context) (PersistenceOutcome, error) { return outcome, err })
}

func (r *PersistenceRunner) worker() {
	defer r.workers.Done()
	for queued := range r.jobs {
		<-queued.scheduledDone
		<-queued.preparedDone
		r.run(queued)
	}
}

func (r *PersistenceRunner) prepareJob(traceCtx context.Context, queued *queuedJob) {
	detached := context.Background()
	if traceCtx != nil {
		detached = trace.ContextWithSpanContext(detached, trace.SpanContextFromContext(traceCtx))
	}
	jobCtx, cancel := context.WithTimeout(detached, r.timeout)
	stop := context.AfterFunc(r.baseCtx, cancel)
	if r.baseCtx.Err() != nil {
		cancel()
	}

	// Report cancellation even when Run cannot unwind yet. Keep this worker and
	// its resources occupied until Run actually exits; never detach the work.
	var terminal sync.Once
	job := queued.job
	report := job.Report
	job.Report = func(status, reason string, failOpen bool, cause error) {
		terminal.Do(func() { report(status, reason, failOpen, cause) })
	}
	reported := make(chan struct{})
	stopReport := context.AfterFunc(jobCtx, func() {
		defer close(reported)
		<-queued.scheduledDone
		reportPersistenceResult(jobCtx, job, PersistenceOutcome{}, nil)
	})
	queued.ctx, queued.job = jobCtx, job
	queued.finish = func() {
		if !stopReport() {
			<-reported
		}
		stop()
		cancel()
	}
}

func (r *PersistenceRunner) run(queued *queuedJob) {
	defer queued.finish()
	jobCtx, job := queued.ctx, queued.job
	defer func() {
		if recovered := recover(); recovered != nil {
			logging.ComponentErrorEvent("memory", "persistence_panic", map[string]interface{}{
				"panic": recovered,
				"stack": string(debug.Stack()),
			})
			reportPersistenceResult(jobCtx, job, PersistenceOutcome{
				Status: "store_failed", Reason: "panic", FailOpen: true,
			}, nil)
		}
	}()

	if jobCtx.Err() != nil {
		reportPersistenceResult(jobCtx, job, PersistenceOutcome{}, nil)
		return
	}
	outcome, err := job.Run(jobCtx)
	reportPersistenceResult(jobCtx, job, outcome, err)
}

func reportPersistenceResult(jobCtx context.Context, job PersistenceJob, outcome PersistenceOutcome, err error) {
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
	r.retireOnce.Do(func() { r.retireErr = r.retireAndWait(grace) })
	return r.retireErr
}

// Done closes only after retirement and the exit of every job and reporter.
// A shutdown deadline is not permission to close resources: owners must defer
// their cleanup until Done, including when RetireAndWait returns an error.
func (r *PersistenceRunner) Done() <-chan struct{} { return r.done }

func (r *PersistenceRunner) retireAndWait(grace time.Duration) error {
	if grace <= 0 {
		grace = DefaultPersistenceShutdownGrace
	}

	r.mu.Lock()
	r.retired = true
	close(r.jobs)
	r.mu.Unlock()

	go func() {
		r.workers.Wait()
		r.cancel()
		close(r.done)
	}()

	select {
	case <-r.done:
		r.cancel()
		return nil
	case <-time.After(grace):
	}

	r.cancel()
	unwound := false
	select {
	case <-r.done:
		unwound = true
	case <-time.After(persistenceCancelUnwind):
	}
	logging.ComponentWarnEvent("memory", "persistence_shutdown_deadline", map[string]interface{}{
		"grace_seconds": grace.Seconds(),
		"unwound":       unwound,
	})
	return ErrPersistenceShutdownDeadline
}
