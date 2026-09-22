package fallback

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v2"
)

func TestFallbackPolicyValidate(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*FallbackPolicy)
		errSubstr string
	}{
		{
			name: "valid default policy",
			mutate: func(p *FallbackPolicy) {
				// default policy is valid
			},
		},
		{
			name: "valid enabled policy",
			mutate: func(p *FallbackPolicy) {
				p.Enabled = true
				p.MaxAttempts = 3
			},
		},
		{
			name: "invalid version",
			mutate: func(p *FallbackPolicy) {
				p.Version = 2
			},
			errSubstr: "unsupported fallback policy version 2",
		},
		{
			name: "zero max_attempts when enabled",
			mutate: func(p *FallbackPolicy) {
				p.Enabled = true
				p.MaxAttempts = 0
			},
			errSubstr: "fallback max_attempts must be >= 1 when enabled",
		},
		{
			name: "negative max_attempts when disabled",
			mutate: func(p *FallbackPolicy) {
				p.Enabled = false
				p.MaxAttempts = -1
			},
			errSubstr: "fallback max_attempts cannot be negative",
		},
		{
			name: "negative total timeout",
			mutate: func(p *FallbackPolicy) {
				p.TotalTimeout = -1 * time.Second
			},
			errSubstr: "fallback total_timeout cannot be negative",
		},
		{
			name: "negative per attempt timeout",
			mutate: func(p *FallbackPolicy) {
				p.PerAttemptTimeout = -1 * time.Second
			},
			errSubstr: "fallback per_attempt_timeout cannot be negative",
		},
		{
			name: "per attempt timeout exceeds total timeout",
			mutate: func(p *FallbackPolicy) {
				p.TotalTimeout = 5 * time.Second
				p.PerAttemptTimeout = 10 * time.Second
			},
			errSubstr: "per_attempt_timeout (10s) cannot exceed total_timeout (5s)",
		},
		{
			name: "invalid status code low",
			mutate: func(p *FallbackPolicy) {
				p.RetryableStatusCodes = []int{99}
			},
			errSubstr: "invalid retryable status code 99",
		},
		{
			name: "invalid status code high",
			mutate: func(p *FallbackPolicy) {
				p.RetryableStatusCodes = []int{600}
			},
			errSubstr: "invalid retryable status code 600",
		},
		{
			name: "negative circuit breaker consecutive failures",
			mutate: func(p *FallbackPolicy) {
				p.CircuitBreaker.ConsecutiveFailures = -1
			},
			errSubstr: "consecutive_failures cannot be negative",
		},
		{
			name: "negative circuit breaker cooldown",
			mutate: func(p *FallbackPolicy) {
				p.CircuitBreaker.CooldownPeriod = -1 * time.Second
			},
			errSubstr: "cooldown_period cannot be negative",
		},
		{
			name: "negative circuit breaker half open probes",
			mutate: func(p *FallbackPolicy) {
				p.CircuitBreaker.HalfOpenProbes = -1
			},
			errSubstr: "half_open_probes cannot be negative",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := DefaultPolicy()
			tt.mutate(&p)
			err := p.Validate()
			if tt.errSubstr == "" {
				if err != nil {
					t.Fatalf("expected valid policy, got: %v", err)
				}
			} else {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.errSubstr)
				}
				if !strings.Contains(err.Error(), tt.errSubstr) {
					t.Fatalf("expected error containing %q, got: %v", tt.errSubstr, err)
				}
			}
		})
	}
}

func TestFallbackPolicyUnmarshalJSON(t *testing.T) {
	t.Run("duration strings", func(t *testing.T) {
		raw := `{"version": 1, "enabled": true, "max_attempts": 4, "total_timeout": "25s", "per_attempt_timeout": "5s", "retryable_status_codes": [502, 503], "circuit_breaker": {"cooldown_period": "45s", "consecutive_failures": 5}}`
		var p FallbackPolicy
		if err := json.Unmarshal([]byte(raw), &p); err != nil {
			t.Fatalf("failed to unmarshal json: %v", err)
		}
		if p.TotalTimeout != 25*time.Second {
			t.Errorf("expected 25s, got %v", p.TotalTimeout)
		}
		if p.PerAttemptTimeout != 5*time.Second {
			t.Errorf("expected 5s, got %v", p.PerAttemptTimeout)
		}
		if p.CircuitBreaker.CooldownPeriod != 45*time.Second {
			t.Errorf("expected 45s, got %v", p.CircuitBreaker.CooldownPeriod)
		}
		if p.CircuitBreaker.ConsecutiveFailures != 5 {
			t.Errorf("expected 5, got %d", p.CircuitBreaker.ConsecutiveFailures)
		}
	})

	t.Run("duration numeric nanoseconds", func(t *testing.T) {
		raw := `{"version": 1, "enabled": true, "total_timeout": 1000000000, "per_attempt_timeout": 500000000, "circuit_breaker": {"cooldown_period": 2000000000}}`
		var p FallbackPolicy
		if err := json.Unmarshal([]byte(raw), &p); err != nil {
			t.Fatalf("failed to unmarshal numeric durations: %v", err)
		}
		if p.TotalTimeout != 1*time.Second {
			t.Errorf("expected 1s, got %v", p.TotalTimeout)
		}
		if p.PerAttemptTimeout != 500*time.Millisecond {
			t.Errorf("expected 500ms, got %v", p.PerAttemptTimeout)
		}
		if p.CircuitBreaker.CooldownPeriod != 2*time.Second {
			t.Errorf("expected 2s, got %v", p.CircuitBreaker.CooldownPeriod)
		}
	})

	t.Run("invalid duration string error", func(t *testing.T) {
		raw := `{"total_timeout": "not-a-duration"}`
		var p FallbackPolicy
		err := json.Unmarshal([]byte(raw), &p)
		if err == nil {
			t.Fatal("expected error parsing invalid duration, got nil")
		}
	})
}

func TestFallbackPolicyWithDefaults(t *testing.T) {
	sparse := FallbackPolicy{
		Enabled: true,
	}
	filled := sparse.WithDefaults()
	if filled.Version != 1 {
		t.Errorf("expected version 1, got %d", filled.Version)
	}
	if filled.MaxAttempts != 3 {
		t.Errorf("expected max_attempts 3, got %d", filled.MaxAttempts)
	}
	if filled.TotalTimeout != 30*time.Second {
		t.Errorf("expected total_timeout 30s, got %v", filled.TotalTimeout)
	}
	if filled.PerAttemptTimeout != 10*time.Second {
		t.Errorf("expected per_attempt_timeout 10s, got %v", filled.PerAttemptTimeout)
	}
	if len(filled.RetryableStatusCodes) != 3 {
		t.Errorf("expected 3 default retryable status codes, got %v", filled.RetryableStatusCodes)
	}
	if filled.CircuitBreaker.ConsecutiveFailures != 3 {
		t.Errorf("expected 3 consecutive failures, got %d", filled.CircuitBreaker.ConsecutiveFailures)
	}
}

func TestFallbackPolicyInherit(t *testing.T) {
	base := FallbackPolicy{
		Version:              1,
		Enabled:              true,
		MaxAttempts:          5,
		TotalTimeout:         45 * time.Second,
		PerAttemptTimeout:    15 * time.Second,
		RetryableStatusCodes: []int{502, 503},
		CircuitBreaker: CircuitBreakerConfig{
			ConsecutiveFailures: 4,
			CooldownPeriod:      20 * time.Second,
			HalfOpenProbes:      2,
		},
	}

	t.Run("partial override inherits base enabled state", func(t *testing.T) {
		// Enabled is omitted (not explicitly configured)
		override := FallbackPolicy{
			MaxAttempts: 2,
			CircuitBreaker: CircuitBreakerConfig{
				ConsecutiveFailures: 1,
			},
		}

		inherited := override.Inherit(base)
		if !inherited.Enabled {
			t.Errorf("expected inherited policy to preserve base.Enabled=true, got false")
		}
		if inherited.MaxAttempts != 2 {
			t.Errorf("expected overridden max_attempts 2, got %d", inherited.MaxAttempts)
		}
		if inherited.CircuitBreaker.ConsecutiveFailures != 1 {
			t.Errorf("expected overridden consecutive failures 1, got %d", inherited.CircuitBreaker.ConsecutiveFailures)
		}
		if inherited.TotalTimeout != 45*time.Second {
			t.Errorf("expected inherited total_timeout 45s, got %v", inherited.TotalTimeout)
		}
		if inherited.PerAttemptTimeout != 15*time.Second {
			t.Errorf("expected inherited per_attempt_timeout 15s, got %v", inherited.PerAttemptTimeout)
		}
		if len(inherited.RetryableStatusCodes) != 2 {
			t.Errorf("expected inherited retryable status codes, got %v", inherited.RetryableStatusCodes)
		}
		if inherited.CircuitBreaker.CooldownPeriod != 20*time.Second {
			t.Errorf("expected inherited cooldown period 20s, got %v", inherited.CircuitBreaker.CooldownPeriod)
		}
		if inherited.CircuitBreaker.HalfOpenProbes != 2 {
			t.Errorf("expected inherited half open probes 2, got %d", inherited.CircuitBreaker.HalfOpenProbes)
		}
	})

	t.Run("explicit false overrides base enabled true", func(t *testing.T) {
		override := FallbackPolicy{
			MaxAttempts: 2,
		}.WithExplicitEnabled(false)

		inherited := override.Inherit(base)
		if inherited.Enabled {
			t.Errorf("expected explicit false to override base enabled true, got true")
		}
		if inherited.MaxAttempts != 2 {
			t.Errorf("expected overridden max_attempts 2, got %d", inherited.MaxAttempts)
		}
	})

	t.Run("explicit true overrides base enabled false", func(t *testing.T) {
		disabledBase := base
		disabledBase.SetExplicitEnabled(false)

		override := FallbackPolicy{
			MaxAttempts: 2,
		}.WithExplicitEnabled(true)

		inherited := override.Inherit(disabledBase)
		if !inherited.Enabled {
			t.Errorf("expected explicit true to override disabled base, got false")
		}
	})
}

func TestFallbackPolicyUnmarshalYAML(t *testing.T) {
	t.Run("tri-state enabled omitted", func(t *testing.T) {
		raw := `
version: 1
max_attempts: 4
total_timeout: 25s
per_attempt_timeout: 5s
retryable_status_codes:
  - 502
  - 503
circuit_breaker:
  cooldown_period: 45s
  consecutive_failures: 5
`
		var p FallbackPolicy
		if err := yaml.Unmarshal([]byte(raw), &p); err != nil {
			t.Fatalf("failed to unmarshal yaml: %v", err)
		}
		if p.ExplicitEnabled() != nil {
			t.Errorf("expected ExplicitEnabled() to be nil when omitted, got %v", *p.ExplicitEnabled())
		}
		if p.Enabled {
			t.Errorf("expected Enabled to default to false before inherit, got true")
		}
		if p.TotalTimeout != 25*time.Second {
			t.Errorf("expected 25s, got %v", p.TotalTimeout)
		}
		if p.PerAttemptTimeout != 5*time.Second {
			t.Errorf("expected 5s, got %v", p.PerAttemptTimeout)
		}
		if p.CircuitBreaker.CooldownPeriod != 45*time.Second {
			t.Errorf("expected 45s, got %v", p.CircuitBreaker.CooldownPeriod)
		}
		if p.CircuitBreaker.ConsecutiveFailures != 5 {
			t.Errorf("expected 5, got %d", p.CircuitBreaker.ConsecutiveFailures)
		}

		// When inherited into an enabled base, it must inherit enabled: true
		base := DefaultEnabledPolicy()
		inherited := p.Inherit(base)
		if !inherited.Enabled {
			t.Errorf("expected partial recipe override to inherit Enabled=true from base, got false")
		}
	})

	t.Run("tri-state enabled explicitly false", func(t *testing.T) {
		raw := `
version: 1
enabled: false
max_attempts: 2
`
		var p FallbackPolicy
		if err := yaml.Unmarshal([]byte(raw), &p); err != nil {
			t.Fatalf("failed to unmarshal yaml: %v", err)
		}
		if p.ExplicitEnabled() == nil {
			t.Fatal("expected ExplicitEnabled() to be non-nil")
		}
		if *p.ExplicitEnabled() != false {
			t.Errorf("expected ExplicitEnabled() to be false")
		}
		if p.Enabled {
			t.Errorf("expected Enabled to be false")
		}

		// Inheriting into enabled base must retain false!
		base := DefaultEnabledPolicy()
		inherited := p.Inherit(base)
		if inherited.Enabled {
			t.Errorf("expected explicit enabled: false to override base.Enabled=true, got true")
		}
	})

	t.Run("tri-state enabled explicitly true", func(t *testing.T) {
		raw := `
version: 1
enabled: true
max_attempts: 3
`
		var p FallbackPolicy
		if err := yaml.Unmarshal([]byte(raw), &p); err != nil {
			t.Fatalf("failed to unmarshal yaml: %v", err)
		}
		if p.ExplicitEnabled() == nil {
			t.Fatal("expected ExplicitEnabled() to be non-nil")
		}
		if *p.ExplicitEnabled() != true {
			t.Errorf("expected ExplicitEnabled() to be true")
		}
		if !p.Enabled {
			t.Errorf("expected Enabled to be true")
		}

		base := DefaultPolicy() // disabled base
		inherited := p.Inherit(base)
		if !inherited.Enabled {
			t.Errorf("expected explicit enabled: true to override base.Enabled=false, got false")
		}
	})
}
