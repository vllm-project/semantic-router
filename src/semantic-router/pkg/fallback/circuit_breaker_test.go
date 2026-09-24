package fallback

import (
	"testing"
	"time"
)

func TestBackendCircuitBreaker(t *testing.T) {
	cfg := CircuitBreakerConfig{
		ConsecutiveFailures: 3,
		CooldownPeriod:      10 * time.Second,
		HalfOpenProbes:      1,
	}
	cb := NewBackendCircuitBreaker(cfg)
	simulatedTime := time.Date(2026, 9, 16, 12, 0, 0, 0, time.UTC)
	cb.nowFunc = func() time.Time { return simulatedTime }

	backend := "primary-backend"

	t.Run("initially closed", func(t *testing.T) {
		if !cb.Allow(backend) {
			t.Errorf("expected initially closed circuit to allow requests")
		}
		if cb.GetState(backend) != StateClosed {
			t.Errorf("expected StateClosed, got %s", cb.GetState(backend))
		}
	})

	t.Run("opens after threshold failures", func(t *testing.T) {
		cb.RecordFailure(backend)
		cb.RecordFailure(backend)
		if !cb.Allow(backend) {
			t.Errorf("should allow before threshold reached")
		}
		if cb.GetState(backend) != StateClosed {
			t.Errorf("should still be StateClosed, got %s", cb.GetState(backend))
		}

		// 3rd failure reaches threshold
		cb.RecordFailure(backend)
		if cb.GetState(backend) != StateOpen {
			t.Errorf("expected StateOpen after 3 failures, got %s", cb.GetState(backend))
		}
		if cb.Allow(backend) {
			t.Errorf("expected open circuit to disallow requests")
		}
	})

	t.Run("transitions to half-open after cooldown and recovers on success", func(t *testing.T) {
		// Advance time by 5s (still within 10s cooldown)
		simulatedTime = simulatedTime.Add(5 * time.Second)
		if cb.Allow(backend) {
			t.Errorf("should disallow while within cooldown")
		}

		// Advance time past cooldown
		simulatedTime = simulatedTime.Add(6 * time.Second) // total +11s
		if !cb.Allow(backend) {
			t.Errorf("should allow probe request after cooldown")
		}
		if cb.GetState(backend) != StateHalfOpen {
			t.Errorf("expected StateHalfOpen, got %s", cb.GetState(backend))
		}

		// Second probe should be rejected because HalfOpenProbes is 1
		if cb.Allow(backend) {
			t.Errorf("should disallow concurrent probe beyond HalfOpenProbes")
		}

		// Successful probe closes circuit
		cb.RecordSuccess(backend)
		if cb.GetState(backend) != StateClosed {
			t.Errorf("expected StateClosed after success, got %s", cb.GetState(backend))
		}
		if !cb.Allow(backend) {
			t.Errorf("should allow requests after returning to closed")
		}
	})

	t.Run("transitions to half-open and re-opens on probe failure", func(t *testing.T) {
		// Cause 3 failures again
		cb.RecordFailure(backend)
		cb.RecordFailure(backend)
		cb.RecordFailure(backend)
		if cb.GetState(backend) != StateOpen {
			t.Errorf("expected StateOpen, got %s", cb.GetState(backend))
		}

		// Advance past cooldown
		simulatedTime = simulatedTime.Add(15 * time.Second)
		if !cb.Allow(backend) {
			t.Errorf("should allow half-open probe")
		}

		// Failed probe re-opens circuit
		cb.RecordFailure(backend)
		if cb.GetState(backend) != StateOpen {
			t.Errorf("expected StateOpen after failed probe, got %s", cb.GetState(backend))
		}
		if cb.Allow(backend) {
			t.Errorf("should disallow after re-opening")
		}
	})

	t.Run("independent tracking across backends", func(t *testing.T) {
		otherBackend := "fallback-backend"
		if !cb.Allow(otherBackend) {
			t.Errorf("independent backend should be allowed even if first backend is open")
		}
		if cb.GetState(otherBackend) != StateClosed {
			t.Errorf("other backend should be closed, got %s", cb.GetState(otherBackend))
		}
	})

	t.Run("half-open allows fresh probe if previous probe dropped after cooldown", func(t *testing.T) {
		droppedBackend := "dropped-probe-backend"
		cb.RecordFailure(droppedBackend)
		cb.RecordFailure(droppedBackend)
		cb.RecordFailure(droppedBackend)
		if cb.GetState(droppedBackend) != StateOpen {
			t.Fatalf("expected StateOpen")
		}

		// Advance past cooldown to enter half-open
		simulatedTime = simulatedTime.Add(15 * time.Second)
		if !cb.Allow(droppedBackend) {
			t.Fatalf("should allow initial probe")
		}
		if cb.GetState(droppedBackend) != StateHalfOpen {
			t.Fatalf("expected StateHalfOpen")
		}

		// Subsequent probe should be blocked while probe in-flight
		if cb.Allow(droppedBackend) {
			t.Errorf("concurrent probe should be disallowed")
		}

		// Suppose probe hangs / drops and never calls RecordSuccess or RecordFailure.
		// Advance another cooldown period.
		simulatedTime = simulatedTime.Add(12 * time.Second)
		if !cb.Allow(droppedBackend) {
			t.Errorf("expected circuit breaker to allow fresh probe after cooldown rather than deadlocking")
		}
	})

	t.Run("reset clears all states", func(t *testing.T) {
		cb.Reset()
		if cb.GetState(backend) != StateClosed {
			t.Errorf("reset should restore StateClosed")
		}
		if !cb.Allow(backend) {
			t.Errorf("reset should allow requests")
		}
	})
}
