package extproc

import (
	"context"
	"sync"
	"sync/atomic"
)

type lifecyclePhase struct {
	once sync.Once
	done chan struct{}
	err  error
}

func (p *lifecyclePhase) run(ctx context.Context, shutdown func() error) error {
	p.once.Do(func() {
		p.done = make(chan struct{})
		go func() {
			p.err = shutdown()
			close(p.done)
		}()
	})
	if err := waitForLifecycleDone(ctx, p.done); err != nil {
		return err
	}
	return p.err
}

type serverLifecycle struct {
	stopping  atomic.Bool
	serving   lifecyclePhase
	resources lifecyclePhase

	watchMu     sync.Mutex
	watchCancel context.CancelFunc
	watchDone   chan struct{}
	reloads     sync.WaitGroup
}

func (l *serverLifecycle) startWatcher(parent context.Context) (context.Context, func()) {
	ctx, cancel := context.WithCancel(parent)
	done := make(chan struct{})
	l.watchMu.Lock()
	l.watchCancel = cancel
	l.watchDone = done
	if l.stopping.Load() {
		cancel()
	}
	l.watchMu.Unlock()

	return ctx, func() {
		cancel()
		close(done)
		l.watchMu.Lock()
		if l.watchDone == done {
			l.watchCancel = nil
		}
		l.watchMu.Unlock()
	}
}

func (l *serverLifecycle) beginShutdown() {
	l.stopping.Store(true)
	l.watchMu.Lock()
	if l.watchCancel != nil {
		l.watchCancel()
	}
	l.watchMu.Unlock()
}

func (l *serverLifecycle) startReload() (func(), bool) {
	l.watchMu.Lock()
	defer l.watchMu.Unlock()
	if l.stopping.Load() {
		return nil, false
	}
	l.reloads.Add(1)
	return l.reloads.Done, true
}

func (l *serverLifecycle) stopAndWaitForBackgroundWork(ctx context.Context) error {
	l.beginShutdown()
	l.watchMu.Lock()
	done := l.watchDone
	l.watchMu.Unlock()
	if err := waitForLifecycleDone(ctx, done); err != nil {
		return err
	}

	reloadsDone := make(chan struct{})
	go func() {
		l.reloads.Wait()
		close(reloadsDone)
	}()
	return waitForLifecycleDone(ctx, reloadsDone)
}

func waitForLifecycleDone(ctx context.Context, done <-chan struct{}) error {
	if done == nil {
		return nil
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		select {
		case <-done:
			return nil
		default:
			return ctx.Err()
		}
	}
}

func (l *serverLifecycle) isStopping() bool {
	return l.stopping.Load()
}
