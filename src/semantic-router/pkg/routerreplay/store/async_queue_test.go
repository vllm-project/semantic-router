package store

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestRunAsyncOpHonorsContextWhileQueueIsSaturated(t *testing.T) {
	queue := make(chan asyncOp, 1)
	queue <- asyncOp{fn: func() error { return nil }, err: make(chan error, 1)}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	started := time.Now()
	err := runAsyncOp(ctx, queue, func() error { return nil })
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("runAsyncOp() error = %v, want context deadline exceeded", err)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("saturated queue wait took %s, want a bounded return", elapsed)
	}
}

func TestRunAsyncOpDoesNotEnqueueAfterContextCancellation(t *testing.T) {
	queue := make(chan asyncOp, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := runAsyncOp(ctx, queue, func() error { return nil })
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("runAsyncOp() error = %v, want context canceled", err)
	}
	if len(queue) != 0 {
		t.Fatalf("queued operations = %d, want 0", len(queue))
	}
}

func TestRunAsyncOpHonorsContextWhileWriterIsStalled(t *testing.T) {
	queue := make(chan asyncOp, 1)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := runAsyncOp(ctx, queue, func() error { return nil })
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("runAsyncOp() error = %v, want context deadline exceeded", err)
	}
}

func TestRunAsyncOpReturnsWriterResult(t *testing.T) {
	queue := make(chan asyncOp)
	wantErr := errors.New("writer failed")
	go func() {
		op := <-queue
		op.err <- op.fn()
	}()

	if err := runAsyncOp(context.Background(), queue, func() error { return wantErr }); !errors.Is(err, wantErr) {
		t.Fatalf("runAsyncOp() error = %v, want %v", err, wantErr)
	}
}

func TestCloseAsyncWriterIsBoundedWhenQueueCannotDrain(t *testing.T) {
	queue := make(chan asyncOp)
	done := make(chan struct{})
	writerDone := make(chan struct{})
	var backendCloseCalls atomic.Int32
	backendCloseStarted := make(chan struct{})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	started := time.Now()
	err := closeAsyncWriterContext(ctx, queue, done, writerDone, func() error {
		backendCloseCalls.Add(1)
		close(backendCloseStarted)
		return nil
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("closeAsyncWriter() error = %v, want context deadline exceeded", err)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("closeAsyncWriter() took %s, want a bounded return", elapsed)
	}
	select {
	case <-backendCloseStarted:
	case <-time.After(time.Second):
		t.Fatal("backend close was not attempted")
	}
	if backendCloseCalls.Load() != 1 {
		t.Fatalf("backend close calls = %d, want 1", backendCloseCalls.Load())
	}
	select {
	case <-done:
	default:
		t.Fatal("writer stop channel was not closed")
	}
}

func TestCloseAsyncWriterIsBoundedWhenBackendCloseStalls(t *testing.T) {
	queue := make(chan asyncOp)
	done := make(chan struct{})
	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		for {
			select {
			case op := <-queue:
				op.err <- op.fn()
			case <-done:
				return
			}
		}
	}()
	backendCloseStarted := make(chan struct{})
	releaseBackendClose := make(chan struct{})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	err := closeAsyncWriterContext(ctx, queue, done, writerDone, func() error {
		close(backendCloseStarted)
		<-releaseBackendClose
		return nil
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("closeAsyncWriter() error = %v, want context deadline exceeded", err)
	}
	select {
	case <-backendCloseStarted:
	case <-time.After(time.Second):
		t.Fatal("backend close was not attempted")
	}
	close(releaseBackendClose)
}

func TestRunBoundedBackendCloseProtectsSynchronousStores(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	backendCloseStarted := make(chan struct{})
	releaseBackendClose := make(chan struct{})

	err := runBoundedBackendClose(ctx, func() error {
		close(backendCloseStarted)
		<-releaseBackendClose
		return nil
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("runBoundedBackendClose() error = %v, want context deadline exceeded", err)
	}
	select {
	case <-backendCloseStarted:
	case <-time.After(time.Second):
		t.Fatal("backend close was not attempted")
	}
	close(releaseBackendClose)
}
