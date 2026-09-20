//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
)

// API-created services outlive every accepted handler and registered worker,
// including native calls that continue after an HTTP response or cancellation.
// Borrowed router/global services are never added as owners.
type serverOwnedResources struct {
	mu        sync.Mutex
	stopping  bool
	requests  sync.WaitGroup
	owners    []io.Closer
	closeOnce sync.Once
	err       error
}

func newServerOwnedResources(owners []io.Closer) *serverOwnedResources {
	resources := &serverOwnedResources{}
	for _, owner := range owners {
		if owner != nil {
			resources.owners = append(resources.owners, owner)
		}
	}
	return resources
}

type serverOwnedResourcesContextKey struct{}

func (r *serverOwnedResources) handler(next http.Handler) http.Handler {
	if next == nil {
		next = http.DefaultServeMux
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		r.mu.Lock()
		if r.stopping {
			r.mu.Unlock()
			http.Error(w, "API server is shutting down", http.StatusServiceUnavailable)
			return
		}
		r.requests.Add(1)
		r.mu.Unlock()
		defer r.requests.Done()
		ctx := context.WithValue(req.Context(), serverOwnedResourcesContextKey{}, r)
		next.ServeHTTP(w, req.WithContext(ctx))
	})
}

// retainAPIWorker registers before its HTTP handler can finish. Drain closes
// registration under the same lock, so Wait cannot race a late Add.
func retainAPIWorker(ctx context.Context) (func(), bool) {
	r, _ := ctx.Value(serverOwnedResourcesContextKey{}).(*serverOwnedResources)
	if r == nil { // Direct handler invocations do not own a listener or resources.
		return func() {}, true
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.stopping {
		return nil, false
	}
	r.requests.Add(1)
	return r.requests.Done, true
}

func (r *serverOwnedResources) beginDrain() {
	if r == nil {
		return
	}
	// This lock prevents Add after Wait starts, even when net/http has accepted
	// a connection whose handler has not started when shutdown begins.
	r.mu.Lock()
	r.stopping = true
	r.mu.Unlock()
}

func (r *serverOwnedResources) drainAndClose() error {
	if r == nil {
		return nil
	}
	r.closeOnce.Do(func() {
		r.beginDrain()
		r.requests.Wait()
		var errs []error
		for i := len(r.owners) - 1; i >= 0; i-- {
			errs = append(errs, r.owners[i].Close())
		}
		r.mu.Lock()
		r.err = errors.Join(errs...)
		r.mu.Unlock()
	})
	return r.closeError()
}

func (r *serverOwnedResources) closeError() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.err
}
