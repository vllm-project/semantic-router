package extproc

import (
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// routerLease tracks in-flight Process calls against one OpenAIRouter, so a
// reload can wait for them to finish instead of racing Close() against a request
// still using the router's Cache, MemoryStore and the rest.
//
// A mutex rather than a sync.WaitGroup, because a lease must reject an acquire
// the instant it starts retiring: with a WaitGroup that means a flag check
// followed by Add(1), which can land after retire()'s Wait() has begun. That
// violates WaitGroup's documented contract ("calls with a positive delta that
// start when the counter is zero must happen before a Wait") and can panic the
// process.
type routerLease struct {
	router *OpenAIRouter

	mu       sync.Mutex
	inFlight int
	retiring bool
	// drained is closed once the lease is retiring and its last in-flight call
	// has released. Created up front so retire() can select on it directly.
	drained chan struct{}
}

func newRouterLease(router *OpenAIRouter) *routerLease {
	return &routerLease{
		router:  router,
		drained: make(chan struct{}),
	}
}

// acquire admits one in-flight call against this lease's router. It returns
// false once the lease has started retiring, and the caller should then retry
// against the RouterService's latest lease.
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

// retire stops admitting new calls and waits up to drainTimeout for in-flight
// calls to finish, so the caller can close the router's resources afterward
// without racing them. Safe to call more than once.
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
