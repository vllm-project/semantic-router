package classification

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// circuitBreakerState represents the three states of the circuit breaker
// state machine.
type circuitBreakerState int

const (
	circuitBreakerClosed   circuitBreakerState = 0
	circuitBreakerOpen     circuitBreakerState = 1
	circuitBreakerHalfOpen circuitBreakerState = 2
)

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

	threshold     int
	openInterval  time.Duration
	maxProbes     int

	breakerState circuitBreakerState
	failureCount int
	openAt       time.Time
	probeCount   int
}

func newCircuitBreaker(cfg *config.RemoteClassifierCircuitBreakerConfig) *circuitBreaker {
	return &circuitBreaker{
		threshold:    cfg.EffectiveConsecutiveFailures(),
		openInterval: time.Duration(cfg.EffectiveOpenInterval()) * time.Millisecond,
		maxProbes:    cfg.EffectiveHalfOpenMaxRequests(),
	}
}

// allow reports whether the caller may issue a request. It transitions
// open→half-open when the interval expires.
func (cb *circuitBreaker) allow() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.breakerState {
	case circuitBreakerClosed:
		return true
	case circuitBreakerOpen:
		if time.Since(cb.openAt) >= cb.openInterval {
			cb.breakerState = circuitBreakerHalfOpen
			cb.probeCount = 0
		}
	}
	if cb.breakerState == circuitBreakerHalfOpen && cb.probeCount < cb.maxProbes {
		cb.probeCount++
		return true
	}
	return false
}

// recordFailure increments the failure count and trips the breaker when the
// threshold is reached. In half-open a single failure returns to open.
func (cb *circuitBreaker) recordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.breakerState {
	case circuitBreakerClosed:
		cb.failureCount++
		if cb.failureCount >= cb.threshold {
			cb.tripLocked()
		}
	case circuitBreakerHalfOpen:
		cb.tripLocked()
	}
}

func (cb *circuitBreaker) tripLocked() {
	cb.breakerState = circuitBreakerOpen
	cb.openAt = time.Now()
	cb.failureCount = 0
	cb.probeCount = 0
}

// recordSuccess resets failures and transitions half-open→closed.
func (cb *circuitBreaker) recordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount = 0
	if cb.breakerState == circuitBreakerHalfOpen {
		cb.breakerState = circuitBreakerClosed
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
		cb:    newCircuitBreaker(cfg),
		name:  name,
	}
}

func (b *circuitBreakingBackend) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	if !b.cb.allow() {
		recordCircuitBreakerSkip(b.name)
		return SequenceClassificationResult{}, &ErrCircuitBreakerOpen{
			Name:       b.name,
			RetryAfter: b.cb.retryAfter().Format(time.RFC3339),
		}
	}

	result, err := b.inner.Classify(ctx, text)
	if err != nil {
		b.cb.recordFailure()
		recordCircuitBreakerFailure(b.name)
		return SequenceClassificationResult{}, err
	}
	b.cb.recordSuccess()
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
		cb:    newCircuitBreaker(cfg),
		name:  name,
	}
}

func (b *circuitBreakingScoringBackend) Score(ctx context.Context, text string) (float64, error) {
	if !b.cb.allow() {
		recordCircuitBreakerSkip(b.name)
		return 0, &ErrCircuitBreakerOpen{
			Name:       b.name,
			RetryAfter: b.cb.retryAfter().Format(time.RFC3339),
		}
	}

	score, err := b.inner.Score(ctx, text)
	if err != nil {
		b.cb.recordFailure()
		recordCircuitBreakerFailure(b.name)
		return 0, err
	}
	b.cb.recordSuccess()
	return score, nil
}