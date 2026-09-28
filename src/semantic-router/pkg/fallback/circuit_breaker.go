package fallback

import (
	"sync"
	"time"
)

// CircuitState represents the current state of a backend circuit breaker.
type CircuitState string

const (
	StateClosed   CircuitState = "closed"
	StateOpen     CircuitState = "open"
	StateHalfOpen CircuitState = "half_open"
)

// BackendCircuitBreaker tracks failure rates and implements backoff per backend target.
type BackendCircuitBreaker struct {
	mu       sync.RWMutex
	cfg      CircuitBreakerConfig
	backends map[string]*backendBreakerState
	nowFunc  func() time.Time
}

type backendBreakerState struct {
	state               CircuitState
	consecutiveFailures int
	lastFailure         time.Time
	lastProbeAt         time.Time
	halfOpenProbes      int
}

// NewBackendCircuitBreaker creates a circuit breaker with the given configuration.
func NewBackendCircuitBreaker(cfg CircuitBreakerConfig) *BackendCircuitBreaker {
	return &BackendCircuitBreaker{
		cfg:      cfg,
		backends: make(map[string]*backendBreakerState),
		nowFunc:  time.Now,
	}
}

// Config returns a copy of the circuit breaker configuration.
func (cb *BackendCircuitBreaker) Config() CircuitBreakerConfig {
	if cb == nil {
		return CircuitBreakerConfig{}
	}
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.cfg
}

// Allow reports whether an attempt is permitted against the specified backend.
func (cb *BackendCircuitBreaker) Allow(backend string) bool {
	if backend == "" {
		return true
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()

	state, exists := cb.backends[backend]
	if !exists {
		return true
	}

	now := cb.nowFunc()
	switch state.state {
	case StateClosed:
		return true
	case StateOpen:
		if now.Sub(state.lastFailure) >= cb.cfg.CooldownPeriod {
			state.state = StateHalfOpen
			state.halfOpenProbes = 1
			state.lastProbeAt = now
			return true
		}
		return false
	case StateHalfOpen:
		// If cooldown has elapsed since the last probe without a recorded outcome,
		// allow a fresh probe to prevent permanent deadlock if a probe hangs or drops.
		if now.Sub(state.lastProbeAt) >= cb.cfg.CooldownPeriod {
			state.halfOpenProbes = 1
			state.lastProbeAt = now
			return true
		}
		if state.halfOpenProbes < cb.cfg.HalfOpenProbes {
			state.halfOpenProbes++
			state.lastProbeAt = now
			return true
		}
		return false
	default:
		return true
	}
}

// RecordSuccess clears consecutive failure counters and restores a backend to closed state.
func (cb *BackendCircuitBreaker) RecordSuccess(backend string) {
	if backend == "" {
		return
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()

	state, exists := cb.backends[backend]
	if !exists {
		return
	}
	state.consecutiveFailures = 0
	state.state = StateClosed
	state.halfOpenProbes = 0
	state.lastProbeAt = time.Time{}
}

// RecordFailure increments failure counters and opens the circuit breaker when thresholds are met.
func (cb *BackendCircuitBreaker) RecordFailure(backend string) {
	if backend == "" {
		return
	}
	cb.mu.Lock()
	defer cb.mu.Unlock()

	state, exists := cb.backends[backend]
	if !exists {
		state = &backendBreakerState{state: StateClosed}
		cb.backends[backend] = state
	}

	state.consecutiveFailures++
	state.lastFailure = cb.nowFunc()
	state.lastProbeAt = time.Time{}

	if state.consecutiveFailures >= cb.cfg.ConsecutiveFailures {
		state.state = StateOpen
		state.halfOpenProbes = 0
	}
}

// GetState returns the current circuit state for a backend (defaults to StateClosed).
func (cb *BackendCircuitBreaker) GetState(backend string) CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if state, ok := cb.backends[backend]; ok {
		return state.state
	}
	return StateClosed
}

// Reset clears all tracked backend states.
func (cb *BackendCircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.backends = make(map[string]*backendBreakerState)
}
