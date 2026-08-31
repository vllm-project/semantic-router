package admission

import (
	"context"
	"sync"
	"time"
)

// Overflow selects what Acquire does when every slot is busy and the wait
// queue is full.
type Overflow string

const (
	// OverflowShed rejects the request with ErrQueueFull.
	OverflowShed Overflow = "shed"
	// OverflowWait blocks past the queue bound until a slot frees or ctx is
	// done.
	OverflowWait Overflow = "wait"
	// OverflowFailOpen admits the request without holding a slot.
	OverflowFailOpen Overflow = "fail_open"
)

// Semaphore is a fixed-slot gate with a bounded wait queue. Queued waiters
// are bounded by the queue capacity and, when set, a queue timeout; the
// request context cancels a wait at any time.
type Semaphore struct {
	slots        chan struct{}
	waiters      chan struct{}
	queueTimeout time.Duration
	overflow     Overflow
}

// NewSemaphore builds a gate with maxConcurrency slots and a wait queue of
// maxQueue requests. A zero queueTimeout waits until the context is done.
func NewSemaphore(maxConcurrency, maxQueue int, queueTimeout time.Duration, overflow Overflow) *Semaphore {
	if maxConcurrency < 1 {
		maxConcurrency = 1
	}
	if maxQueue < 0 {
		maxQueue = 0
	}
	if overflow == "" {
		overflow = OverflowShed
	}
	return &Semaphore{
		slots:        make(chan struct{}, maxConcurrency),
		waiters:      make(chan struct{}, maxQueue),
		queueTimeout: queueTimeout,
		overflow:     overflow,
	}
}

func (s *Semaphore) Acquire(ctx context.Context) (Ticket, error) {
	select {
	case s.slots <- struct{}{}:
		return s.ticket(), nil
	default:
	}

	queued := true
	select {
	case s.waiters <- struct{}{}:
	default:
		switch s.overflow {
		case OverflowFailOpen:
			return func() {}, nil
		case OverflowWait:
			queued = false
		default:
			return nil, ErrQueueFull
		}
	}
	if queued {
		defer func() { <-s.waiters }()
	}

	var timeout <-chan time.Time
	if s.queueTimeout > 0 {
		timer := time.NewTimer(s.queueTimeout)
		defer timer.Stop()
		timeout = timer.C
	}

	select {
	case s.slots <- struct{}{}:
		return s.ticket(), nil
	case <-timeout:
		return nil, ErrQueueFull
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *Semaphore) ticket() Ticket {
	var once sync.Once
	return func() {
		once.Do(func() { <-s.slots })
	}
}
