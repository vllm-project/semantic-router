package extproc

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// routerLease tracks in-flight Process calls against one OpenAIRouter
// generation, so a reload can wait for those calls to finish before closing
// the router's owned resources instead of racing a concurrent Close()
// against an in-flight request's use of Cache/MemoryStore/etc.
type routerLease struct {
	router   *OpenAIRouter
	inFlight sync.WaitGroup
	retiring atomic.Bool
}

func newRouterLease(router *OpenAIRouter) *routerLease {
	return &routerLease{router: router}
}

// acquire admits one in-flight call against this lease's router. It returns
// false once the lease has started retiring, in which case the caller
// should retry against the RouterService's latest lease instead of using
// this one.
func (l *routerLease) acquire() bool {
	if l == nil || l.retiring.Load() {
		return false
	}
	l.inFlight.Add(1)
	if l.retiring.Load() {
		// retire() may have called inFlight.Wait() between our first check
		// above and Add() — undo it so we never leave a call racing a Wait()
		// that already returned and let a request reach a router whose
		// resources retire() is about to close.
		l.inFlight.Done()
		return false
	}
	return true
}

// release matches a prior successful acquire.
func (l *routerLease) release() {
	if l == nil {
		return
	}
	l.inFlight.Done()
}

// retire stops admitting new calls and waits up to drainTimeout for
// in-flight calls to finish, so the caller can safely close the router's
// owned resources afterward without racing them.
func (l *routerLease) retire(drainTimeout time.Duration) {
	if l == nil {
		return
	}
	l.retiring.Store(true)

	done := make(chan struct{})
	go func() {
		l.inFlight.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(drainTimeout):
		logging.ComponentWarnEvent("extproc", "router_lease_drain_timed_out", map[string]interface{}{
			"timeout_seconds": drainTimeout.Seconds(),
		})
	}
}
