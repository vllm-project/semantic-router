package config

import "time"

// ShadowComparisonConfig enables bounded multi-arm shadow comparison: the same
// normalized request is replayed to a fixed set of candidate arms in
// observation mode without altering the user-visible primary route
// (issue #3376). Secrets never live here; the router injects per-arm
// credentials at dispatch time.
type ShadowComparisonConfig struct {
	Enabled   bool              `yaml:"enabled,omitempty"`
	MaxWaitMS int               `yaml:"max_wait_ms,omitempty"`
	Arms      []ShadowArmConfig `yaml:"arms,omitempty"`
}

// ShadowArmConfig describes one candidate model evaluated on real traffic.
//
//   - Model is the model name sent to the arm endpoint.
//   - Endpoint is an OpenAI-compatible base URL for the arm's own backend.
//   - TimeoutSeconds bounds a single arm attempt (0 = default).
type ShadowArmConfig struct {
	Name           string `yaml:"name,omitempty"`
	Model          string `yaml:"model"`
	Endpoint       string `yaml:"endpoint"`
	TimeoutSeconds int    `yaml:"timeout_seconds,omitempty"`
}

// IsEnabled reports whether shadow comparison is wired and has arms to run.
func (c ShadowComparisonConfig) IsEnabled() bool {
	if !c.Enabled || len(c.Arms) == 0 {
		return false
	}
	for _, arm := range c.Arms {
		if arm.Model == "" || arm.Endpoint == "" {
			return false
		}
	}
	return true
}

// DefaultShadowMaxWait bounds the aggregate shadow window when unset.
const DefaultShadowMaxWait = 3 * time.Second

// GetMaxWait returns the total time shadow work may take after the primary
// decision. Arms are best-effort inside this window; the primary response is
// never delayed by it (fire-and-collect, zero blocking).
func (c ShadowComparisonConfig) GetMaxWait() time.Duration {
	if c.MaxWaitMS <= 0 {
		return DefaultShadowMaxWait
	}
	return time.Duration(c.MaxWaitMS) * time.Millisecond
}
