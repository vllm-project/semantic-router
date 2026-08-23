package config

import (
	"fmt"
	"net"
	"regexp"
	"strings"
	"time"
)

const (
	defaultBackendDispatchBindAddress = "0.0.0.0"
	defaultBackendDispatchPort        = 8180
	defaultBackendDispatchAudience    = "vllm-sr.backend-dispatch"
	defaultBackendDispatchTTL         = "30s"
	defaultBackendDispatchMaxBody     = int64(64 << 20)
)

var backendDispatchAudiencePattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,127}$`)

// BackendDispatchConfig defines the private HTTP listener that owns every
// physical model attempt. Deployment networking decides reachability; a
// short-lived request-bound capability authorizes each call.
type BackendDispatchConfig struct {
	BindAddress         string `yaml:"bind_address"`
	Port                int    `yaml:"port"`
	Audience            string `yaml:"audience"`
	CapabilityTTL       string `yaml:"capability_ttl"`
	MaxRequestBodyBytes int64  `yaml:"max_request_body_bytes"`
}

func DefaultBackendDispatchConfig() BackendDispatchConfig {
	return BackendDispatchConfig{
		BindAddress:         defaultBackendDispatchBindAddress,
		Port:                defaultBackendDispatchPort,
		Audience:            defaultBackendDispatchAudience,
		CapabilityTTL:       defaultBackendDispatchTTL,
		MaxRequestBodyBytes: defaultBackendDispatchMaxBody,
	}
}

func (config BackendDispatchConfig) CapabilityLifetime() (time.Duration, error) {
	lifetime, err := time.ParseDuration(config.CapabilityTTL)
	if err != nil || lifetime < time.Second || lifetime > time.Minute || lifetime%time.Millisecond != 0 {
		return 0, fmt.Errorf("global.services.backend_dispatch.capability_ttl must be a whole number of milliseconds between 1s and 1m")
	}
	return lifetime, nil
}

func validateBackendDispatch(config BackendDispatchConfig) error {
	if config.BindAddress != strings.TrimSpace(config.BindAddress) || net.ParseIP(config.BindAddress) == nil {
		return fmt.Errorf("global.services.backend_dispatch.bind_address must be a canonical IP address")
	}
	if config.Port < 1 || config.Port > 65535 {
		return fmt.Errorf("global.services.backend_dispatch.port must be between 1 and 65535")
	}
	if !backendDispatchAudiencePattern.MatchString(config.Audience) {
		return fmt.Errorf("global.services.backend_dispatch.audience is invalid")
	}
	if _, err := config.CapabilityLifetime(); err != nil {
		return err
	}
	if config.MaxRequestBodyBytes < 1<<20 || config.MaxRequestBodyBytes > 256<<20 {
		return fmt.Errorf("global.services.backend_dispatch.max_request_body_bytes must be between 1 MiB and 256 MiB")
	}
	return nil
}
