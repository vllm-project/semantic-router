package config

import (
	"fmt"
	"time"
)

const (
	DefaultRoutingPreviewTimeoutSeconds = 120
	DefaultRoutingPreviewMaxConcurrency = 16
	// RoutingPreviewResponseWriteAllowance leaves time to deliver a timeout response.
	RoutingPreviewResponseWriteAllowance = 5 * time.Second
)

// RoutingPreviewConfig bounds management Preview inference independently of
// other HTTP routes and per-model admission gates.
type RoutingPreviewConfig struct {
	// RequestTimeoutSeconds limits inference after the request body is decoded.
	RequestTimeoutSeconds *int `yaml:"request_timeout_seconds,omitempty" jsonschema:"minimum=1,maximum=3600,default=120"`
	// MaxConcurrency counts workers until inference finishes, even after timeout.
	// Changing this listener-start limit requires a deployment restart.
	MaxConcurrency *int `yaml:"max_concurrency,omitempty" jsonschema:"minimum=1,default=16"`
}

func (c RoutingPreviewConfig) RequestTimeout() time.Duration {
	seconds := DefaultRoutingPreviewTimeoutSeconds
	if c.RequestTimeoutSeconds != nil {
		seconds = *c.RequestTimeoutSeconds
	}
	return time.Duration(seconds) * time.Second
}

func (c RoutingPreviewConfig) ConcurrencyLimit() int {
	if c.MaxConcurrency != nil {
		return *c.MaxConcurrency
	}
	return DefaultRoutingPreviewMaxConcurrency
}

func validateRoutingPreviewConfig(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	c := cfg.API.RoutingPreview
	if c.RequestTimeoutSeconds != nil && (*c.RequestTimeoutSeconds < 1 || *c.RequestTimeoutSeconds > 3600) {
		return fmt.Errorf("global.services.api.routing_preview.request_timeout_seconds must be between 1 and 3600")
	}
	if c.ConcurrencyLimit() < 1 {
		return fmt.Errorf("global.services.api.routing_preview.max_concurrency must be positive")
	}
	return nil
}

// ValidateRoutingPreviewReload keeps the listener's admission gate stable while
// workers from earlier runtime generations may still hold its tickets.
func ValidateRoutingPreviewReload(current, next *RouterConfig) error {
	if current == nil || next == nil {
		return nil
	}
	if current.API.RoutingPreview.ConcurrencyLimit() != next.API.RoutingPreview.ConcurrencyLimit() {
		return fmt.Errorf("global.services.api.routing_preview.max_concurrency changed; activate the candidate through the deployment workflow and restart the management listener")
	}
	return nil
}
