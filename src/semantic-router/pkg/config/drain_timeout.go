package config

import (
	"fmt"
	"time"
)

// DefaultDrainTimeout is how long the standalone gateway lets the requests in
// flight finish once the Router begins to shut down, when
// global.router.drain_timeout_seconds is omitted.
const DefaultDrainTimeout = 25 * time.Second

// MaxDrainTimeoutSeconds bounds global.router.drain_timeout_seconds: a day.
const MaxDrainTimeoutSeconds = 24 * 60 * 60

// CapabilityDrainTimeout is global.router.drain_timeout_seconds, which only
// the standalone gateway honors: behind Envoy, Envoy drains the client
// connections.
const CapabilityDrainTimeout = "drain_timeout"

func init() {
	GatewayCapabilities.MustRegister(CapabilityDrainTimeout, GatewayCapability{
		Modes: []GatewayMode{GatewayStandalone},
		Uses: func(cfg *RouterConfig) []CapabilityUse {
			if cfg.DrainTimeoutSeconds == nil {
				return nil
			}
			const path = "global.router.drain_timeout_seconds"
			return []CapabilityUse{{Path: path, Subject: path}}
		},
	})
}

// DrainTimeout is how long the standalone gateway lets the requests in flight
// finish once the Router begins to shut down: DefaultDrainTimeout where
// global.router.drain_timeout_seconds is unset, and none for an explicit 0.
func (c *RouterConfig) DrainTimeout() time.Duration {
	if c == nil || c.DrainTimeoutSeconds == nil {
		return DefaultDrainTimeout
	}
	return time.Duration(*c.DrainTimeoutSeconds) * time.Second
}

func validateDrainTimeoutContract(cfg *RouterConfig) error {
	if seconds := cfg.DrainTimeoutSeconds; seconds != nil && (*seconds < 0 || *seconds > MaxDrainTimeoutSeconds) {
		return fmt.Errorf("global.router.drain_timeout_seconds must be between 0 and %d, got %d", MaxDrainTimeoutSeconds, *seconds)
	}
	return nil
}
