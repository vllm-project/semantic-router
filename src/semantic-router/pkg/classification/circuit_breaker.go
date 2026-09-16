package classification

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

// circuitBreakerState represents the three states of the circuit breaker
// state machine.
type circuitBreakerState int

const (
	circuitBreakerClosed   circuitBreakerState = 0
	circuitBreakerOpen     circuitBreakerState = 1
	circuitBreakerHalfOpen circuitBreakerState = 2
)

// String returns a stable label for metrics and structured logs.
func (s circuitBreakerState) String() string {
	switch s {
	case circuitBreakerOpen:
		return "open"
	case circuitBreakerHalfOpen:
		return "half_open"
	default:
		return "closed"
	}
}

// ErrCircuitBreakerOpen is returned when a call is skipped because the
// circuit breaker is open. The typed error preserves the existing fail-closed
// contract: decision-tree Unknown, rules.on_unknown, and prompt-guard on_error
// still apply.
type ErrCircuitBreakerOpen struct {
	Name       string
	RetryAfter string
}

func (e *ErrCircuitBreakerOpen) Error() string {
	return fmt.Sprintf("circuit breaker open for %q, retry after %s", e.Name, e.RetryAfter)
}

// circuitBreaker implements the standard closed/open/half-open state machine
// per remote classifier backend instance.
type circuitBreaker struct {
	mu sync.Mutex

	name         string
	threshold    int
	openInterval time.Duration
	maxProbes    int

	breakerState circuitBreakerState
	failureCount int
	openAt       time.Time

	// half-open probe accounting
	admittedProbes   int
	unresolvedProbes int
	halfOpenFailed   bool

	// generation increments each time half-open is entered. An admitted
	// request records the generation it was admitted under; callbacks ignore
	// completions whose generation does not match the current one, preventing
	// stale closed-state responses from interfering with half-open probes.
	halfOpenGen int
}

func newCircuitBreaker(name string, cfg *config.RemoteClassifierCircuitBreakerConfig) *circuitBreaker {
	cb := &circuitBreaker{
		name:         name,
		threshold:    cfg.EffectiveConsecutiveFailures(),
		openInterval: time.Duration(cfg.EffectiveOpenInterval()) * time.Millisecond,
		maxProbes:    cfg.EffectiveHalfOpenMaxRequests(),
		breakerState: circuitBreakerClosed,
	}
	cb.recordStateLocked(circuitBreakerClosed)
	return cb
}

// allow reports whether the caller may issue a request. It transitions
// open→half-open when the interval expires and returns the admission
// generation for matching completions to their admission state.
func (cb *circuitBreaker) allow() (bool, int) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.breakerState {
	case circuitBreakerClosed:
		return true, 0
	case circuitBreakerOpen:
		if time.Since(cb.openAt) >= cb.openInterval {
			cb.setHalfOpenLocked()
		}
	}
	if cb.breakerState == circuitBreakerHalfOpen && cb.admittedProbes < cb.maxProbes {
		cb.admittedProbes++
		cb.unresolvedProbes++
		return true, cb.halfOpenGen
	}
	return false, 0
}

func (cb *circuitBreaker) setHalfOpenLocked() {
	from := cb.breakerState
	cb.breakerState = circuitBreakerHalfOpen
	cb.admittedProbes = 0
	cb.unresolvedProbes = 0
	cb.halfOpenFailed = false
	cb.halfOpenGen++
	cb.transitionLocked(from, circuitBreakerHalfOpen)
}

// recordFailure increments the failure count and trips the breaker when the
// threshold is reached. In half-open a failure is marked and the breaker
// reopens once all admitted probes resolve. admissionGen is the generation
// returned by allow; completions from a stale generation are ignored.
func (cb *circuitBreaker) recordFailure(admissionGen int) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.breakerState {
	case circuitBreakerClosed:
		cb.failureCount++
		if cb.failureCount >= cb.threshold {
			cb.tripLocked()
		}
	case circuitBreakerHalfOpen:
		if admissionGen != cb.halfOpenGen {
			return
		}
		if cb.unresolvedProbes > 0 {
			cb.unresolvedProbes--
		}
		cb.halfOpenFailed = true
		cb.resolveHalfOpenLocked()
	}
}

func (cb *circuitBreaker) tripLocked() {
	from := cb.breakerState
	cb.breakerState = circuitBreakerOpen
	cb.openAt = time.Now()
	cb.failureCount = 0
	cb.admittedProbes = 0
	cb.unresolvedProbes = 0
	cb.halfOpenFailed = false
	cb.transitionLocked(from, circuitBreakerOpen)
}

// recordSuccess resets failures and transitions half-open→closed when all
// admitted probes have resolved without failure. admissionGen is the
// generation returned by allow; completions from a stale generation are
// ignored.
func (cb *circuitBreaker) recordSuccess(admissionGen int) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount = 0
	if cb.breakerState != circuitBreakerHalfOpen {
		return
	}
	if admissionGen != cb.halfOpenGen {
		return
	}
	if cb.unresolvedProbes > 0 {
		cb.unresolvedProbes--
	}
	cb.resolveHalfOpenLocked()
}

// releaseProbe releases an admitted half-open probe slot without counting it
// as a success or failure. Used when an error is not counted toward breaker
// state (e.g. caller cancellation). In closed/open state it is a no-op.
// admissionGen must be the generation returned by allow.
func (cb *circuitBreaker) releaseProbe(admissionGen int) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.breakerState != circuitBreakerHalfOpen {
		return
	}
	if admissionGen != cb.halfOpenGen {
		return
	}
	if cb.unresolvedProbes > 0 {
		cb.unresolvedProbes--
	}
	cb.resolveHalfOpenLocked()
}

// resolveHalfOpenLocked completes a half-open round when no unresolved probes
// remain. If any probe failed, the breaker reopens; otherwise it closes.
func (cb *circuitBreaker) resolveHalfOpenLocked() {
	if cb.unresolvedProbes > 0 {
		return
	}
	if cb.halfOpenFailed {
		cb.tripLocked()
	} else {
		from := cb.breakerState
		cb.breakerState = circuitBreakerClosed
		cb.failureCount = 0
		cb.transitionLocked(from, circuitBreakerClosed)
	}
}

// getState returns the current state (for testing).
func (cb *circuitBreaker) getState() circuitBreakerState {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.breakerState
}

// retryAfter returns the time when the breaker will allow half-open probes.
func (cb *circuitBreaker) retryAfter() time.Time {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.openAt.Add(cb.openInterval)
}

// transitionLocked emits the observability surface for a single state-machine
// transition. Callers must hold cb.mu.
func (cb *circuitBreaker) transitionLocked(from, to circuitBreakerState) {
	cb.recordStateLocked(to)
	recordCircuitBreakerTransition(cb.name, from, to)
}

// recordStateLocked publishes the current state to the metrics gauge. Callers
// must hold cb.mu.
func (cb *circuitBreaker) recordStateLocked(state circuitBreakerState) {
	circuitBreakerStateGauge.WithLabelValues(cb.name).Set(float64(state))
}

// circuitBreakingBackend wraps a SequenceClassifierBackend with a circuit
// breaker. The wrapper is transparent to callers when the breaker is closed.
type circuitBreakingBackend struct {
	inner SequenceClassifierBackend
	cb    *circuitBreaker
	name  string
}

// newCircuitBreakingBackend wraps inner with a circuit breaker when the
// config has the breaker enabled. Returns inner unchanged when disabled.
func newCircuitBreakingBackend(inner SequenceClassifierBackend, cfg *config.RemoteClassifierCircuitBreakerConfig, name string) SequenceClassifierBackend {
	if cfg == nil || !cfg.Enabled {
		return inner
	}
	return &circuitBreakingBackend{
		inner: inner,
		cb:    newCircuitBreaker(name, cfg),
		name:  name,
	}
}

func (b *circuitBreakingBackend) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	allowed, gen := b.cb.allow()
	if !allowed {
		recordCircuitBreakerSkip(b.name)
		return SequenceClassificationResult{}, &ErrCircuitBreakerOpen{
			Name:       b.name,
			RetryAfter: b.cb.retryAfter().Format(time.RFC3339),
		}
	}

	result, err := b.inner.Classify(ctx, text)
	if err != nil {
		if isUnavailableError(err) {
			b.cb.recordFailure(gen)
			recordCircuitBreakerFailure(b.name)
		} else {
			b.cb.releaseProbe(gen)
		}
		return SequenceClassificationResult{}, err
	}
	b.cb.recordSuccess(gen)
	return result, nil
}

// circuitBreakingScoringBackend wraps a ScoringBackend with a circuit breaker.
type circuitBreakingScoringBackend struct {
	inner ScoringBackend
	cb    *circuitBreaker
	name  string
}

func newCircuitBreakingBackendScoring(inner ScoringBackend, cfg *config.RemoteClassifierCircuitBreakerConfig, name string) ScoringBackend {
	if cfg == nil || !cfg.Enabled {
		return inner
	}
	return &circuitBreakingScoringBackend{
		inner: inner,
		cb:    newCircuitBreaker(name, cfg),
		name:  name,
	}
}

func (b *circuitBreakingScoringBackend) Score(ctx context.Context, text string) (float64, error) {
	allowed, gen := b.cb.allow()
	if !allowed {
		recordCircuitBreakerSkip(b.name)
		return 0, &ErrCircuitBreakerOpen{
			Name:       b.name,
			RetryAfter: b.cb.retryAfter().Format(time.RFC3339),
		}
	}

	score, err := b.inner.Score(ctx, text)
	if err != nil {
		if isUnavailableError(err) {
			b.cb.recordFailure(gen)
			recordCircuitBreakerFailure(b.name)
		} else {
			b.cb.releaseProbe(gen)
		}
		return 0, err
	}
	b.cb.recordSuccess(gen)
	return score, nil
}

// isUnavailableError reports whether err is a retry-exhausted connector error
// indicating the backend is unreachable or returning a retryable status.
// Caller cancellation is never counted. Classifier-owned deadline exceeded
// (backend timeout) IS counted as unavailable.
func isUnavailableError(err error) bool {
	connErr := new(connector.Error)
	if errors.As(err, &connErr) {
		switch connErr.Kind {
		case connector.KindTransport:
			if connErr.Retryable {
				return true
			}
			// Non-retryable transport after budget exhausted.
			// Caller cancellation → not counted.
			if errors.Is(connErr.Cause, context.Canceled) {
				return false
			}
			// Classifier-owned DeadlineExceeded (backend timeout) → counted.
			// Other non-retryable transport → not counted.
			return errors.Is(connErr.Cause, context.DeadlineExceeded)
		case connector.KindStatus:
			return connErr.Retryable
		}
		return false
	}
	// Bare context errors (not wrapped by connector) are caller-driven.
	return false
}
