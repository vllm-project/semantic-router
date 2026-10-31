package config

import "fmt"

// ClassifierCircuitBreaker is opt-in. Pointers distinguish omitted
// defaults from explicit invalid zero values without mutating canonical input.
type ClassifierCircuitBreaker struct {
	Enabled             bool `yaml:"enabled" json:"enabled"`
	ConsecutiveFailures *int `yaml:"consecutive_failures,omitempty" json:"consecutive_failures,omitempty"`
	OpenIntervalMs      *int `yaml:"open_interval_ms,omitempty" json:"open_interval_ms,omitempty"`
}

func (c *ClassifierCircuitBreaker) Validate() error {
	if c == nil || !c.Enabled {
		return nil
	}
	if c.ConsecutiveFailures != nil && *c.ConsecutiveFailures <= 0 {
		return fmt.Errorf("circuit_breaker.consecutive_failures must be positive")
	}
	if c.OpenIntervalMs != nil && *c.OpenIntervalMs <= 0 {
		return fmt.Errorf("circuit_breaker.open_interval_ms must be positive")
	}
	return nil
}

// EffectiveValues returns defaults only when enabled: three failures, 30s open.
func (c *ClassifierCircuitBreaker) EffectiveValues() (failures, intervalMs int) {
	if c == nil || !c.Enabled {
		return 0, 0
	}
	failures, intervalMs = 3, 30000
	if c.ConsecutiveFailures != nil {
		failures = *c.ConsecutiveFailures
	}
	if c.OpenIntervalMs != nil {
		intervalMs = *c.OpenIntervalMs
	}
	return
}

func validateExternalClassifierCircuitBreakers(cfg *RouterConfig) error {
	for _, external := range cfg.ExternalModels {
		if err := external.CircuitBreaker.Validate(); err != nil {
			return fmt.Errorf("global.model_catalog.external[%q]: %w", external.Name, err)
		}
	}
	return nil
}
