package connector

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// CircuitBreakerOptions enables a per-client breaker. Nil or zero disables it.
// Name is a static backend/operation label, never an endpoint or request value.
type CircuitBreakerOptions struct {
	Name                string
	ConsecutiveFailures int
	OpenInterval        time.Duration
}

type breakerState uint8

const (
	breakerClosed breakerState = iota
	breakerOpen
	breakerHalfOpen
)

func (s breakerState) String() string { return [...]string{"closed", "open", "half_open"}[s] }

type breakerTicket struct {
	generation uint64
	probe      bool
}

type circuitBreaker struct {
	mu         sync.Mutex
	options    CircuitBreakerOptions
	now        func() time.Time
	state      breakerState
	generation uint64
	failures   int
	openedAt   time.Time
	probing    bool
}

func newCircuitBreaker(options *CircuitBreakerOptions) (*circuitBreaker, error) {
	if options == nil || *options == (CircuitBreakerOptions{}) {
		return nil, nil
	}
	if options.Name == "" || options.ConsecutiveFailures <= 0 || options.OpenInterval <= 0 {
		return nil, fmt.Errorf("connector circuit breaker requires a static name and positive threshold and interval")
	}
	b := &circuitBreaker{options: *options, now: time.Now}
	return b, nil
}

func (b *circuitBreaker) transition(state breakerState) {
	previous := b.state
	b.state = state
	b.generation++
	b.failures, b.probing = 0, false
	if state == breakerOpen {
		b.openedAt = b.now()
	}
	metrics.RemoteConnectorBreakerTransitions.WithLabelValues(b.options.Name, state.String()).Inc()
	logging.ComponentEvent("connector", "circuit_breaker_transition", map[string]interface{}{
		"backend": b.options.Name, "from": previous.String(), "to": state.String(),
	})
}

func (b *circuitBreaker) admit() (breakerTicket, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == breakerOpen && !b.now().Before(b.openedAt.Add(b.options.OpenInterval)) {
		b.transition(breakerHalfOpen)
	}
	if b.state == breakerOpen || b.probing {
		metrics.RemoteConnectorBreakerSkipped.WithLabelValues(b.options.Name).Inc()
		return breakerTicket{}, false
	}
	ticket := breakerTicket{generation: b.generation, probe: b.state == breakerHalfOpen}
	b.probing = ticket.probe
	return ticket, true
}

// Completion belongs to its admission generation. An old closed-state call
// cannot resolve the single recovery probe, even if it spans the cooldown.
func (b *circuitBreaker) complete(ticket breakerTicket, ctx context.Context, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if ticket.generation != b.generation {
		return
	}
	ctx = callerContext(ctx)
	success := ctx != nil && ctx.Err() == nil && err == nil
	failure := false
	var remote *Error
	if ctx != nil && ctx.Err() == nil && errors.As(err, &remote) {
		failure = remote.isAvailabilityFailure()
	}
	if ticket.probe {
		b.probing = false
		if success {
			b.transition(breakerClosed)
		} else if failure {
			b.transition(breakerOpen)
		}
		return
	}
	if success {
		b.failures = 0
	} else if failure {
		b.failures++
		if b.failures >= b.options.ConsecutiveFailures {
			b.transition(breakerOpen)
		}
	}
}

type callerContextKey struct{}

// WithCallerContext records the original caller before an adapter adds its own
// deadline. The breaker uses it only for failure attribution; cancellation and
// deadlines still follow ctx. Nested adapters retain the original caller.
func WithCallerContext(ctx context.Context) context.Context {
	return context.WithValue(ctx, callerContextKey{}, callerContext(ctx))
}

func callerContext(ctx context.Context) context.Context {
	if ctx != nil {
		if caller, ok := ctx.Value(callerContextKey{}).(context.Context); ok {
			return caller
		}
	}
	return ctx
}

// CircuitBreakerEnabled reports whether this client has an enabled breaker.
func (c *Client) CircuitBreakerEnabled() bool { return c != nil && c.breaker != nil }
