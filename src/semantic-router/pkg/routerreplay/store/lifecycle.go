package store

import (
	"context"
	"errors"
	"sync"
)

var ErrStorageClosed = errors.New("router replay storage is closing or closed")

// storageLifecycle prevents a replay store from accepting a mutation after
// shutdown begins. Close takes the exclusive lock, so every operation that
// already acquired a lease completes before the writer queue and client are
// drained. Repeated and concurrent Close calls observe the same result.
type storageLifecycle struct {
	mu        sync.RWMutex
	closeOnce sync.Once
	closeDone chan struct{}
	closed    bool
	closeErr  error
}

func newStorageLifecycle() *storageLifecycle {
	return &storageLifecycle{closeDone: make(chan struct{})}
}

func (l *storageLifecycle) beginMutation() (func(), error) {
	if l == nil {
		return func() {}, nil
	}
	l.mu.RLock()
	if l.closed {
		l.mu.RUnlock()
		return nil, ErrStorageClosed
	}
	return l.mu.RUnlock, nil
}

func (l *storageLifecycle) close(closeStore func() error) error {
	if l == nil {
		ctx, cancel := context.WithTimeout(context.Background(), asyncStoreCloseTimeout)
		defer cancel()
		return runBoundedBackendClose(ctx, closeStore)
	}
	l.closeOnce.Do(func() {
		l.mu.Lock()
		l.closed = true
		ctx, cancel := context.WithTimeout(context.Background(), asyncStoreCloseTimeout)
		l.closeErr = runBoundedBackendClose(ctx, closeStore)
		cancel()
		l.mu.Unlock()
		close(l.closeDone)
	})
	<-l.closeDone
	return l.closeErr
}

func (l *storageLifecycle) closeContext(ctx context.Context, closeStore func() error) error {
	if l == nil {
		return runBoundedBackendClose(ctx, closeStore)
	}
	l.closeOnce.Do(func() {
		l.mu.Lock()
		l.closed = true
		l.closeErr = runBoundedBackendClose(ctx, closeStore)
		l.mu.Unlock()
		close(l.closeDone)
	})
	<-l.closeDone
	return l.closeErr
}
