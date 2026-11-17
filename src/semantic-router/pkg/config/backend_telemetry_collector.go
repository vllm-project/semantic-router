package config

import (
	"fmt"
	"strings"
	"time"
)

// Backend telemetry collection defaults.
const (
	DefaultBackendTelemetryPollInterval   = 2 * time.Second
	DefaultBackendTelemetryRequestTimeout = time.Second
)

// BackendTelemetryConfig controls the off-request-path collector that reads
// serving-engine metrics from backends that name an engine. It is disabled by
// default; enabling it changes no routing on its own.
type BackendTelemetryConfig struct {
	Enabled bool `yaml:"enabled,omitempty"`
	// PollInterval is how often each backend is scraped (default 2s).
	PollInterval string `yaml:"poll_interval,omitempty"`
	// RequestTimeout bounds one scrape of one backend (default 1s).
	RequestTimeout string `yaml:"request_timeout,omitempty"`
	// TTL is how long an observation stays usable; it defaults to three poll
	// intervals so one missed scrape does not discard a backend.
	TTL string `yaml:"ttl,omitempty"`
}

// BackendTelemetrySettings are the parsed collector settings.
type BackendTelemetrySettings struct {
	PollInterval   time.Duration
	RequestTimeout time.Duration
	TTL            time.Duration
}

// Settings parses the collector durations and applies their defaults.
func (c BackendTelemetryConfig) Settings() (BackendTelemetrySettings, error) {
	settings := BackendTelemetrySettings{
		PollInterval:   DefaultBackendTelemetryPollInterval,
		RequestTimeout: DefaultBackendTelemetryRequestTimeout,
	}
	for _, field := range []struct {
		name string
		raw  string
		out  *time.Duration
	}{
		{"poll_interval", c.PollInterval, &settings.PollInterval},
		{"request_timeout", c.RequestTimeout, &settings.RequestTimeout},
		{"ttl", c.TTL, &settings.TTL},
	} {
		if strings.TrimSpace(field.raw) == "" {
			continue
		}
		d, err := time.ParseDuration(field.raw)
		if err != nil || d <= 0 {
			return BackendTelemetrySettings{}, fmt.Errorf(
				"global.services.observability.backend_telemetry.%s must be a positive duration such as 2s", field.name,
			)
		}
		*field.out = d
	}
	if settings.TTL == 0 {
		settings.TTL = 3 * settings.PollInterval
	}
	if settings.TTL < settings.PollInterval {
		return BackendTelemetrySettings{}, fmt.Errorf(
			"global.services.observability.backend_telemetry.ttl must not be shorter than poll_interval",
		)
	}
	return settings, nil
}

func validateBackendTelemetryConfig(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	_, err := cfg.Observability.BackendTelemetry.Settings()
	return err
}
