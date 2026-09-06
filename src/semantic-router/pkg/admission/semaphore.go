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
	// OverflowWait blocks for a queue slot instead of shedding; queued work
	// never exceeds the queue bound.
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
	if overflow == OverflowWait && maxQueue < 1 {
		maxQueue = 1
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

	var timeout <-chan time.Time
	if s.queueTimeout > 0 {
		timer := time.NewTimer(s.queueTimeout)
		defer timer.Stop()
		timeout = timer.C
	}

	overflowTicket, err := s.enterQueue(ctx, timeout)
	if err != nil || overflowTicket != nil {
		return overflowTicket, err
	}
	defer func() { <-s.waiters }()

	select {
	case s.slots <- struct{}{}:
		return s.ticket(), nil
	case <-timeout:
		return nil, ErrQueueFull
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// enterQueue takes a queue slot, applying the overflow policy when the queue
// is full. A nil, nil return means the caller holds a queue slot; a non-nil
// ticket admits a fail_open caller without one.
func (s *Semaphore) enterQueue(ctx context.Context, timeout <-chan time.Time) (Ticket, error) {
	select {
	case s.waiters <- struct{}{}:
		return nil, nil
	default:
	}
	switch s.overflow {
	case OverflowFailOpen:
		return func() {}, nil
	case OverflowWait:
		select {
		case s.waiters <- struct{}{}:
			return nil, nil
		case <-timeout:
			return nil, ErrQueueFull
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	default:
		return nil, ErrQueueFull
	}
}

func (s *Semaphore) ticket() Ticket {
	var once sync.Once
	return func() {
		once.Do(func() { <-s.slots })
	}
}
