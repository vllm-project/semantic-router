package routerreplay

import (
	"context"
	"maps"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const DefaultOutcomeQueueCapacity = 256

// Outcome draining has its own fixed budget, independent of memory persistence
// configuration and the timeout of an individual replay storage operation.
const outcomeShutdownGrace = 5 * time.Second

// ShareOutcomeQueue shares one bounded outcome writer across a replay runtime.
// Call only during construction, before any recorder is used. The writer starts
// lazily; recipe count does not multiply the runtime's workers or queue capacity.
func ShareOutcomeQueue(recorders ...*Recorder) {
	queue := newOutcomeQueue(DefaultOutcomeQueueCapacity, outcomeShutdownGrace)
	for _, recorder := range recorders {
		if recorder != nil {
			recorder.outcomes = queue
		}
	}
}

// TryAppendOutcome reports queue acceptance, not durable storage success.
// It never waits for storage or capacity. AppendOutcome remains synchronous.
func (r *Recorder) TryAppendOutcome(id string, outcome Outcome) bool {
	outcome.Metadata = maps.Clone(outcome.Metadata)
	return r.outcomes.submit(queuedOutcome{recorder: r, id: id, outcome: outcome})
}

// DrainOutcomes stops acceptance and drains the shared runtime queue. Call after
// producers retire and before closing any of the runtime's replay stores.
func (r *Recorder) DrainOutcomes() error {
	return r.outcomes.close()
}

type queuedOutcome struct {
	recorder *Recorder
	id       string
	outcome  Outcome
}

type outcomeQueue struct {
	mu        sync.Mutex
	started   bool
	closed    bool
	events    chan queuedOutcome
	done      chan struct{}
	ctx       context.Context
	cancel    context.CancelFunc
	grace     time.Duration
	closeOnce sync.Once
	closeErr  error
}

func newOutcomeQueue(capacity int, grace time.Duration) *outcomeQueue {
	if grace <= 0 {
		grace = outcomeShutdownGrace
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &outcomeQueue{
		events: make(chan queuedOutcome, capacity), done: make(chan struct{}),
		ctx: ctx, cancel: cancel, grace: grace,
	}
}

func (q *outcomeQueue) submit(event queuedOutcome) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return false
	}
	if !q.started {
		q.started = true
		go q.run()
	}
	select {
	case q.events <- event:
		return true
	default:
		return false
	}
}

func (q *outcomeQueue) run() {
	defer close(q.done)
	for event := range q.events {
		if q.ctx.Err() != nil {
			metrics.RecordPluginExecution("router_replay_outcome", "", "dropped", 0)
			continue
		}
		q.write(event)
	}
}

func (q *outcomeQueue) write(event queuedOutcome) {
	defer func() {
		if recovered := recover(); recovered != nil {
			metrics.RecordPluginExecution("router_replay_outcome", "", "failed", 0)
			logging.ComponentErrorEvent("router_replay", "outcome_write_panic", map[string]interface{}{"replay_id": event.id, "panic": recovered})
		}
	}()
	if err := event.recorder.AppendOutcomeContext(q.ctx, event.id, event.outcome); err != nil {
		metrics.RecordPluginExecution("router_replay_outcome", "", "failed", 0)
		logging.ComponentWarnEvent("router_replay", "outcome_write_failed", map[string]interface{}{
			"replay_id": event.id, "target_ref": event.outcome.TargetRef, "verdict": event.outcome.Verdict, "error": err.Error(),
		})
	}
}

func (q *outcomeQueue) close() error {
	q.closeOnce.Do(func() {
		q.mu.Lock()
		q.closed = true
		close(q.events)
		if !q.started {
			close(q.done)
		}
		q.mu.Unlock()
		defer q.cancel()
		timer := time.NewTimer(q.grace)
		defer timer.Stop()
		select {
		case <-q.done:
		case <-timer.C:
			q.closeErr = context.DeadlineExceeded
		}
	})
	return q.closeErr
}
