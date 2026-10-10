package memory

import (
	"context"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// defaultStoreCloseWait bounds how long Close blocks. Work still running after
// that keeps the client alive; it is released once the work stops.
const defaultStoreCloseWait = 10 * time.Second

// storeLifecycle admits store operations until close. Close cancels them and
// releases the client only after every operation and background job has
// returned, so no call runs against a released client. Zero value is ready.
type storeLifecycle struct {
	mu        sync.Mutex
	closed    bool
	ops       sync.WaitGroup
	jobs      sync.WaitGroup
	stopCtx   context.Context
	stop      context.CancelFunc
	closeWait time.Duration
}

func (l *storeLifecycle) stopContextLocked() context.Context {
	if l.stopCtx == nil {
		l.stopCtx, l.stop = context.WithCancel(context.Background())
	}
	return l.stopCtx
}

// begin admits one operation. The returned context is also cancelled when Close
// starts; the caller must use it and must run release.
func (l *storeLifecycle) begin(ctx context.Context, enabled bool) (context.Context, func(), error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return ctx, nil, ErrStoreClosed
	}
	if !enabled {
		return ctx, nil, ErrStoreDisabled
	}
	l.ops.Add(1)
	opCtx, cancel := context.WithCancel(ctx)
	unhook := context.AfterFunc(l.stopContextLocked(), cancel)
	var once sync.Once
	return opCtx, func() {
		once.Do(func() {
			unhook()
			cancel()
			l.ops.Done()
		})
	}, nil
}

func (l *storeLifecycle) isClosed() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.closed
}

// goBackground runs job on a context that Close cancels; Close waits for it.
func (l *storeLifecycle) goBackground(job func(context.Context)) {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return
	}
	ctx := l.stopContextLocked()
	l.jobs.Add(1)
	l.mu.Unlock()
	go func() {
		defer l.jobs.Done()
		job(ctx)
	}()
}

// close stops admitting work, cancels it, and runs release once every admitted
// operation and job has returned. It runs only once; later calls return nil.
func (l *storeLifecycle) close(backend string, release func() error) error {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return nil
	}
	l.closed = true
	if l.stop != nil {
		l.stop()
	}
	wait := l.closeWait
	l.mu.Unlock()
	if wait <= 0 {
		wait = defaultStoreCloseWait
	}

	drained := make(chan struct{})
	go func() {
		l.ops.Wait()
		l.jobs.Wait()
		close(drained)
	}()
	select {
	case <-drained:
		return release()
	case <-time.After(wait):
		logging.Warnf("Memory %s store: work still running %s after Close; the client is released when it stops", backend, wait)
		go func() {
			<-drained
			if err := release(); err != nil {
				logging.Warnf("Memory %s store: deferred client close failed: %v", backend, err)
			}
		}()
		return nil
	}
}
