package proxy

import (
	"context"
	"net/http"
	"sync/atomic"
	"time"
)

// ServeWithLiveAuthorization keeps an established proxy response bound to the
// session and permissions that authorized its request. A check runs before
// every response write and flush; the timer also closes an idle upstream stream
// after revocation, even when it has no next chunk to write.
func ServeWithLiveAuthorization(
	w http.ResponseWriter,
	r *http.Request,
	next http.Handler,
	check func(context.Context) error,
	pollInterval time.Duration,
) {
	if next == nil || check == nil {
		http.Error(w, "Authorization is unavailable", http.StatusServiceUnavailable)
		return
	}
	if err := check(r.Context()); err != nil {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	if pollInterval <= 0 {
		pollInterval = 250 * time.Millisecond
	}

	ctx, cancel := context.WithCancel(r.Context())
	writer := &liveAuthorizationWriter{
		ResponseWriter: w,
		ctx:            ctx,
		cancel:         cancel,
		check:          check,
	}
	monitorDone := make(chan struct{})
	go func() {
		defer close(monitorDone)
		ticker := time.NewTicker(pollInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if err := check(ctx); err != nil {
					writer.revoke()
					return
				}
			}
		}
	}()
	defer func() {
		cancel()
		<-monitorDone
	}()

	next.ServeHTTP(writer, r.WithContext(ctx))
}

type liveAuthorizationWriter struct {
	http.ResponseWriter
	ctx        context.Context
	cancel     context.CancelFunc
	check      func(context.Context) error
	revoked    atomic.Bool
	headerSent atomic.Bool
}

func (w *liveAuthorizationWriter) revoke() {
	if w.revoked.CompareAndSwap(false, true) {
		w.cancel()
	}
}

func (w *liveAuthorizationWriter) authorized() bool {
	if w.revoked.Load() {
		return false
	}
	if err := w.check(w.ctx); err != nil {
		w.revoke()
		return false
	}
	return !w.revoked.Load()
}

func (w *liveAuthorizationWriter) denyBeforeHeader() {
	if w.headerSent.CompareAndSwap(false, true) {
		http.Error(w.ResponseWriter, "Forbidden", http.StatusForbidden)
	}
}

func (w *liveAuthorizationWriter) WriteHeader(status int) {
	if !w.authorized() {
		w.denyBeforeHeader()
		return
	}
	if w.headerSent.CompareAndSwap(false, true) {
		w.ResponseWriter.WriteHeader(status)
	}
}

func (w *liveAuthorizationWriter) Write(p []byte) (int, error) {
	if !w.authorized() {
		w.denyBeforeHeader()
		// ReverseProxy may already have sent HTTP 200. Consume the chunk so it
		// cannot be emitted and let the canceled request close the upstream.
		return len(p), nil
	}
	w.headerSent.Store(true)
	return w.ResponseWriter.Write(p)
}

func (w *liveAuthorizationWriter) Flush() {
	if !w.authorized() {
		w.denyBeforeHeader()
		return
	}
	w.headerSent.Store(true)
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *liveAuthorizationWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}
