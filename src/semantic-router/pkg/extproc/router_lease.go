package extproc

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// routerLease tracks in-flight Process calls against one OpenAIRouter
// generation, so a reload can wait for those calls to finish before closing
// the router's owned resources instead of racing a concurrent Close()
// against an in-flight request's use of Cache/MemoryStore/etc.
//
// Admission and draining are guarded by one mutex rather than a
// sync.WaitGroup: a lease must reject an acquire the instant it starts
// retiring, and doing that with a WaitGroup means checking a flag and then
// calling Add(1), which can land after retire()'s Wait() has begun. That
// violates WaitGroup's documented contract ("calls with a positive delta
// that start when the counter is zero must happen before a Wait") and can
// panic the process. Holding mu across the flag check and the counter
// increment makes the two atomic with respect to retire().
type routerLease struct {
	router *OpenAIRouter

	mu       sync.Mutex
	inFlight int
	retiring bool
	// drained is closed once the lease is retiring and its last in-flight
	// call has released. It is created up front so retire() can select on it
	// without further synchronization.
	drained chan struct{}
}

func newRouterLease(router *OpenAIRouter) *routerLease {
	return &routerLease{
		router:  router,
		drained: make(chan struct{}),
	}
}

// acquire admits one in-flight call against this lease's router. It returns
// false once the lease has started retiring, in which case the caller
// should retry against the RouterService's latest lease instead of using
// this one.
func (l *routerLease) acquire() bool {
	if l == nil {
		return false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.retiring {
		return false
	}
	l.inFlight++
	return true
}

// release matches a prior successful acquire.
func (l *routerLease) release() {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.inFlight--
	if l.retiring && l.inFlight == 0 {
		l.markDrainedLocked()
	}
}

// markDrainedLocked closes drained at most once. Callers must hold mu.
func (l *routerLease) markDrainedLocked() {
	select {
	case <-l.drained:
	default:
		close(l.drained)
	}
}

// retire stops admitting new calls and waits up to drainTimeout for
// in-flight calls to finish, so the caller can safely close the router's
// owned resources afterward without racing them. It is safe to call more
// than once.
func (l *routerLease) retire(drainTimeout time.Duration) {
	if l == nil {
		return
	}
	l.mu.Lock()
	l.retiring = true
	if l.inFlight == 0 {
		l.markDrainedLocked()
	}
	drained := l.drained
	l.mu.Unlock()

	select {
	case <-drained:
	case <-time.After(drainTimeout):
		logging.ComponentWarnEvent("extproc", "router_lease_drain_timed_out", map[string]interface{}{
			"timeout_seconds": drainTimeout.Seconds(),
		})
	}
}
