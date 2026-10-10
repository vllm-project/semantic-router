//go:build !windows

package apiserver

import (
	"context"
	"errors"
)

var errAPIWorkerUnavailable = errors.New("API runtime worker is unavailable")

type retainedAPIResult[T any] struct {
	response T
	err      error
}

// runRetainedAPIWork separates request cancellation from runtime ownership.
// The HTTP caller stops waiting at its deadline; the model call ends with the
// same context, and the work keeps its leases and server drain registration
// until it has returned. The caller supplies its domain's already-acquired
// leases and admission ticket.
func runRetainedAPIWork[T any](ctx context.Context, release func(), invoke func(context.Context) (T, error)) (T, error) {
	var zero T
	if err := ctx.Err(); err != nil {
		release()
		return zero, err
	}
	workerDone, ok := retainAPIWorker(ctx)
	if !ok {
		release()
		return zero, errAPIWorkerUnavailable
	}
	completed := make(chan retainedAPIResult[T], 1)
	go func() {
		var result retainedAPIResult[T]
		defer func() {
			if recover() != nil {
				result = retainedAPIResult[T]{err: errAPIWorkerUnavailable}
			}
			release()
			workerDone()
			completed <- result
		}()
		if result.err = ctx.Err(); result.err == nil {
			result.response, result.err = invoke(ctx)
		}
	}()
	select {
	case <-ctx.Done():
		return zero, ctx.Err()
	case result := <-completed:
		// A deadline wins over a late successful return.
		if err := ctx.Err(); err != nil {
			return zero, err
		}
		return result.response, result.err
	}
}
