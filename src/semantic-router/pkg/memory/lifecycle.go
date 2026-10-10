package memory

import (
	"context"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// storeCloseWait bounds how long Close waits for in-flight work before it
// releases the client anyway; every store call already carries its own timeout.
const storeCloseWait = 10 * time.Second

// storeLifecycle admits store operations until close, and lets Close wait for
// in-flight operations and background jobs before the store releases its client.
// The zero value is ready to use.
type storeLifecycle struct {
	mu     sync.Mutex
	closed bool
	ops    sync.WaitGroup
	jobs   sync.WaitGroup
	jobCtx context.Context
	stop   context.CancelFunc
}

// begin admits one operation; the caller must run the returned release.
func (l *storeLifecycle) begin(enabled bool) (func(), error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return nil, ErrStoreClosed
	}
	if !enabled {
		return nil, ErrStoreDisabled
	}
	l.ops.Add(1)
	return l.ops.Done, nil
}

func (l *storeLifecycle) isClosed() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.closed
}

// goBackground runs job on a context that close cancels; close waits for it.
func (l *storeLifecycle) goBackground(job func(context.Context)) {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return
	}
	if l.jobCtx == nil {
		l.jobCtx, l.stop = context.WithCancel(context.Background())
	}
	ctx := l.jobCtx
	l.jobs.Add(1)
	l.mu.Unlock()
	go func() {
		defer l.jobs.Done()
		job(ctx)
	}()
}

// close stops admitting work, cancels background jobs, and waits (bounded) for
// both. It reports false when the store was already closed.
func (l *storeLifecycle) close(backend string) bool {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return false
	}
	l.closed = true
	if l.stop != nil {
		l.stop()
	}
	l.mu.Unlock()

	drained := make(chan struct{})
	go func() {
		l.ops.Wait()
		l.jobs.Wait()
		close(drained)
	}()
	select {
	case <-drained:
	case <-time.After(storeCloseWait):
		logging.Warnf("Memory %s store: closing with work still in flight after %s", backend, storeCloseWait)
	}
	return true
}
