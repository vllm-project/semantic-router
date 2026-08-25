package accesspublisher

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

const (
	defaultWorkerIdleDelay  = 250 * time.Millisecond
	defaultWorkerMinBackoff = 100 * time.Millisecond
	defaultWorkerMaxBackoff = 5 * time.Second
)

// Processor is the single-iteration seam implemented by Engine. Keeping the
// loop outside Engine preserves its transaction/state-machine responsibility
// and makes process lifecycle independently testable.
type Processor interface {
	ProcessOnce(context.Context) (ProcessResult, error)
}

type WorkerOptions struct {
	Processor  Processor
	IdleDelay  time.Duration
	MinBackoff time.Duration
	MaxBackoff time.Duration
}

// Worker continuously drains publication work while exposing its current
// health to Router readiness. Transient store failures do not require a
// process restart: readiness fails closed until a later successful iteration
// proves that both stores are reachable again.
type Worker struct {
	processor  Processor
	idleDelay  time.Duration
	minBackoff time.Duration
	maxBackoff time.Duration

	mu            sync.RWMutex
	started       bool
	lastErr       error
	startedSignal chan struct{}
	startOnce     sync.Once
}

func NewWorker(options WorkerOptions) (*Worker, error) {
	if options.Processor == nil {
		return nil, errors.New("routing publication processor is required")
	}
	if options.IdleDelay == 0 {
		options.IdleDelay = defaultWorkerIdleDelay
	}
	if options.MinBackoff == 0 {
		options.MinBackoff = defaultWorkerMinBackoff
	}
	if options.MaxBackoff == 0 {
		options.MaxBackoff = defaultWorkerMaxBackoff
	}
	if options.IdleDelay < time.Millisecond || options.IdleDelay > time.Minute ||
		options.MinBackoff < time.Millisecond || options.MinBackoff > time.Minute ||
		options.MaxBackoff < options.MinBackoff || options.MaxBackoff > time.Minute {
		return nil, errors.New("routing publication worker delays are invalid")
	}
	return &Worker{
		processor: options.Processor, idleDelay: options.IdleDelay,
		minBackoff: options.MinBackoff, maxBackoff: options.MaxBackoff,
		startedSignal: make(chan struct{}),
	}, nil
}

// Started is closed after Run has installed its readiness state. Process
// composition can wait on this signal without polling or racing the worker
// goroutine.
func (worker *Worker) Started() <-chan struct{} {
	if worker == nil || worker.startedSignal == nil {
		closed := make(chan struct{})
		close(closed)
		return closed
	}
	return worker.startedSignal
}

func (worker *Worker) Ready(context.Context) error {
	if worker == nil || worker.processor == nil {
		return errors.New("routing publication worker is unavailable")
	}
	worker.mu.RLock()
	defer worker.mu.RUnlock()
	if !worker.started {
		return errors.New("routing publication worker has not started")
	}
	if worker.lastErr != nil {
		return fmt.Errorf("routing publication worker is unhealthy: %w", worker.lastErr)
	}
	return nil
}

func (worker *Worker) Run(ctx context.Context) error {
	if worker == nil || worker.processor == nil {
		return errors.New("routing publication worker is unavailable")
	}
	worker.setStarted()
	backoff := worker.minBackoff
	for {
		result, err := worker.processor.ProcessOnce(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			worker.setError(err)
			if err := waitWorker(ctx, backoff); err != nil {
				return err
			}
			backoff *= 2
			if backoff > worker.maxBackoff {
				backoff = worker.maxBackoff
			}
			continue
		}
		worker.setError(nil)
		backoff = worker.minBackoff
		if result.Disposition != ProcessNoWork && result.Disposition != ProcessWaitingForAcks {
			continue
		}
		if err := waitWorker(ctx, worker.idleDelay); err != nil {
			return err
		}
	}
}

func (worker *Worker) setStarted() {
	worker.mu.Lock()
	worker.started = true
	worker.lastErr = nil
	worker.mu.Unlock()
	worker.startOnce.Do(func() { close(worker.startedSignal) })
}

func (worker *Worker) setError(err error) {
	worker.mu.Lock()
	worker.lastErr = err
	worker.mu.Unlock()
}

func waitWorker(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

var _ Processor = (*Engine)(nil)
