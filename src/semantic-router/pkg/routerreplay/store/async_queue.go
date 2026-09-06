package store

import (
	"context"
	"errors"
	"time"
)

const asyncStoreCloseTimeout = 5 * time.Second

type asyncOp struct {
	fn  func() error
	err chan error
}

func enqueueAsyncOp(ctx context.Context, queue chan<- asyncOp, op asyncOp) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case queue <- op:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func waitAsyncOp(ctx context.Context, result <-chan error) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case err := <-result:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func runAsyncOp(ctx context.Context, queue chan<- asyncOp, fn func() error) error {
	result := make(chan error, 1)
	if err := enqueueAsyncOp(ctx, queue, asyncOp{fn: fn, err: result}); err != nil {
		return err
	}
	return waitAsyncOp(ctx, result)
}

func closeAsyncWriter(
	queue chan<- asyncOp,
	done chan struct{},
	writerDone <-chan struct{},
	closeBackend func() error,
) error {
	ctx, cancel := context.WithTimeout(context.Background(), asyncStoreCloseTimeout)
	defer cancel()
	return closeAsyncWriterContext(ctx, queue, done, writerDone, closeBackend)
}

func closeAsyncWriterContext(
	ctx context.Context,
	queue chan<- asyncOp,
	done chan struct{},
	writerDone <-chan struct{},
	closeBackend func() error,
) error {
	drainErr := runAsyncOp(ctx, queue, func() error { return nil })
	close(done)
	var writerErr error
	select {
	case <-writerDone:
	case <-ctx.Done():
		writerErr = ctx.Err()
	}
	return errors.Join(drainErr, writerErr, runBoundedBackendClose(ctx, closeBackend))
}

func runBoundedBackendClose(ctx context.Context, closeBackend func() error) error {
	result := make(chan error, 1)
	go func() {
		result <- closeBackend()
	}()
	select {
	case err := <-result:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}
