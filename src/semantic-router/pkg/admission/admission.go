// Package admission bounds concurrent Router Model inference per model
// deployment. An Admissioner hands out tickets; callers hold a ticket for the
// duration of one forward pass and release it when done.
package admission

import (
	"context"
	"errors"
)

// ErrQueueFull reports that the gate shed the request because every slot is
// busy and the wait queue is at capacity or timed out.
var ErrQueueFull = errors.New("admission queue full")

// Ticket releases the slot held by a successful Acquire. It is safe to call
// more than once.
type Ticket func()

// Admissioner grants admission to run one model inference. Acquire blocks
// until a slot frees, the queue policy sheds the request, or ctx is done.
type Admissioner interface {
	Acquire(ctx context.Context) (Ticket, error)
}

// Do runs fn under an admission ticket and releases it afterwards.
func Do[T any](ctx context.Context, admissioner Admissioner, fn func() (T, error)) (T, error) {
	ticket, err := admissioner.Acquire(ctx)
	if err != nil {
		var zero T
		return zero, err
	}
	defer ticket()
	return fn()
}

// Noop admits every request immediately. It is the default gate and preserves
// unbounded behavior at zero cost.
type Noop struct{}

func (Noop) Acquire(context.Context) (Ticket, error) {
	return func() {}, nil
}
