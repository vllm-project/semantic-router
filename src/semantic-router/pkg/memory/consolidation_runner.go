package memory

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	DefaultConsolidationCooldown      = 60 * time.Second
	DefaultConsolidationTimeout       = 30 * time.Second
	DefaultConsolidationConcurrency   = 1
	DefaultConsolidationShutdownGrace = 5 * time.Second
	consolidationCancelUnwind         = time.Second
)

// ErrConsolidationShutdownDeadline is returned when in-flight consolidation
// does not exit within the shutdown grace plus a short cancel unwind.
var ErrConsolidationShutdownDeadline = errors.New("memory consolidation shutdown deadline exceeded")

// ConsolidationOptions configures a background consolidation runner.
// Non-positive cooldown, timeout, and concurrency use the defaults above.
type ConsolidationOptions struct {
	Cooldown    time.Duration
	Timeout     time.Duration
	Concurrency int
}

// ConsolidationRunner merges lexically similar memories for one user after a
// successful write. Enqueue is non-blocking. A nil runner is safe to call.
type ConsolidationRunner struct {
	store    Store
	cooldown time.Duration
	timeout  time.Duration
	slots    chan struct{}

	baseCtx context.Context
	cancel  context.CancelFunc

	mu           sync.Mutex
	retired      bool
	inflight     map[string]struct{}
	lastAccepted map[string]time.Time

	wg         sync.WaitGroup
	done       chan struct{}
	retireOnce sync.Once
	retireErr  error
}

// NewConsolidationRunner starts no workers. Each accepted user occupies one
// concurrency slot until ConsolidateUser returns.
func NewConsolidationRunner(store Store, opts ConsolidationOptions) *ConsolidationRunner {
	if opts.Cooldown <= 0 {
		opts.Cooldown = DefaultConsolidationCooldown
	}
	if opts.Timeout <= 0 {
		opts.Timeout = DefaultConsolidationTimeout
	}
	if opts.Concurrency < 1 {
		opts.Concurrency = DefaultConsolidationConcurrency
	}
	ctx, cancel := context.WithCancel(context.Background())
	runner := &ConsolidationRunner{
		store:        store,
		cooldown:     opts.Cooldown,
		timeout:      opts.Timeout,
		slots:        make(chan struct{}, opts.Concurrency),
		baseCtx:      ctx,
		cancel:       cancel,
		inflight:     make(map[string]struct{}),
		lastAccepted: make(map[string]time.Time),
		done:         make(chan struct{}),
	}
	logging.ComponentEvent("memory", "consolidation_runner_started", map[string]interface{}{
		"cooldown_seconds": opts.Cooldown.Seconds(),
		"timeout_seconds":  opts.Timeout.Seconds(),
		"concurrency":      opts.Concurrency,
	})
	return runner
}

// Enqueue accepts at most one consolidation for userID per cooldown window.
// A run already in flight, a full concurrency limit, or retirement is skipped
// without blocking the caller.
func (r *ConsolidationRunner) Enqueue(userID string) {
	if r == nil || userID == "" || r.store == nil {
		return
	}
	r.mu.Lock()
	if r.retired {
		r.mu.Unlock()
		recordConsolidation("skipped", "shutting_down", 0, 0)
		return
	}
	if _, ok := r.inflight[userID]; ok {
		r.mu.Unlock()
		recordConsolidation("skipped", "in_flight", 0, 0)
		return
	}
	r.evictExpiredLastAcceptedLocked(time.Now())
	if last, ok := r.lastAccepted[userID]; ok && time.Since(last) < r.cooldown {
		r.mu.Unlock()
		recordConsolidation("skipped", "cooldown", 0, 0)
		return
	}
	select {
	case r.slots <- struct{}{}:
	default:
		r.mu.Unlock()
		recordConsolidation("rejected", "concurrency_full", 0, 0)
		return
	}
	r.inflight[userID] = struct{}{}
	r.lastAccepted[userID] = time.Now()
	r.wg.Add(1)
	r.mu.Unlock()
	go r.run(userID)
}

// evictExpiredLastAcceptedLocked drops cooldown entries older than the cooldown
// window so a long-lived router does not retain every processed user ID forever.
// Caller must hold r.mu.
func (r *ConsolidationRunner) evictExpiredLastAcceptedLocked(now time.Time) {
	for id, acceptedAt := range r.lastAccepted {
		if now.Sub(acceptedAt) >= r.cooldown {
			delete(r.lastAccepted, id)
		}
	}
}

func (r *ConsolidationRunner) run(userID string) {
	defer func() {
		<-r.slots
		r.mu.Lock()
		delete(r.inflight, userID)
		r.mu.Unlock()
		r.wg.Done()
	}()
	defer func() {
		if recovered := recover(); recovered != nil {
			recordConsolidation("failed", "panic", 0, 0)
			logging.ComponentErrorEvent("memory", "consolidation_panic", map[string]interface{}{
				"panic": fmt.Sprint(recovered),
			})
		}
	}()

	ctx, cancel := context.WithTimeout(r.baseCtx, r.timeout)
	defer cancel()
	merged, deleted, err := ConsolidateUser(ctx, r.store, userID)
	switch {
	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		recordConsolidation("timeout", "consolidate_timeout", 0, 0)
	case errors.Is(ctx.Err(), context.Canceled):
		recordConsolidation("cancelled", "shutdown", 0, 0)
	case err != nil:
		recordConsolidation("failed", "consolidate_error", 0, 0)
		logging.ComponentWarnEvent("memory", "consolidation_failed", map[string]interface{}{
			"reason":      "consolidate_error",
			"error_class": fmt.Sprintf("%T", err),
		})
	default:
		recordConsolidation("completed", "finished", merged, deleted)
	}
}

// RetireAndWait rejects new work and waits for accepted runs to finish.
// After grace it cancels in-flight work. Done stays open until those runs exit.
func (r *ConsolidationRunner) RetireAndWait(grace time.Duration) error {
	if r == nil {
		return nil
	}
	r.retireOnce.Do(func() { r.retireErr = r.retire(grace) })
	return r.retireErr
}

// Done closes after retirement and the exit of every accepted run.
func (r *ConsolidationRunner) Done() <-chan struct{} {
	if r == nil {
		closed := make(chan struct{})
		close(closed)
		return closed
	}
	return r.done
}

func (r *ConsolidationRunner) retire(grace time.Duration) error {
	if grace <= 0 {
		grace = DefaultConsolidationShutdownGrace
	}
	r.mu.Lock()
	r.retired = true
	r.mu.Unlock()

	go func() {
		r.wg.Wait()
		r.cancel()
		close(r.done)
	}()

	select {
	case <-r.done:
		return nil
	case <-time.After(grace):
	}
	r.cancel()

	select {
	case <-r.done:
		return nil
	case <-time.After(consolidationCancelUnwind):
	}
	logging.ComponentWarnEvent("memory", "consolidation_shutdown_deadline", map[string]interface{}{
		"grace_seconds": grace.Seconds(),
	})
	return ErrConsolidationShutdownDeadline
}

func recordConsolidation(status, reason string, merged, deleted int) {
	RecordMemoryConsolidation(status, reason, merged, deleted)
	logging.ComponentEvent("memory", "consolidation_finished", map[string]interface{}{
		"status":  status,
		"reason":  reason,
		"merged":  merged,
		"deleted": deleted,
	})
}
