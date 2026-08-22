package config

import (
	"fmt"
	"net"
	"os"
	"strings"
)

const (
	ManagementInternalListenerEnv = "VLLM_SR_MANAGEMENT_INTERNAL_LISTENER"

	ManagementAuthModeBearer   = "bearer"
	ManagementAuthModeDisabled = "disabled"

	ManagementPermWildcard = "*"
)

// ManagementAPIConfig configures the router management HTTP listener (#2463).
// Listener on/off is controlled by the legacy -enable-api flag at startup, not
// by a config enabled field.
type ManagementAPIConfig struct {
	BindAddress    string                  `yaml:"bind_address,omitempty"`
	Port           int                     `yaml:"port,omitempty"`
	RemoteExposure bool                    `yaml:"remote_exposure,omitempty"`
	Auth           ManagementAPIAuthConfig `yaml:"auth,omitempty"`
}

// ManagementAPIAuthConfig controls authentication for non-health management routes.
type ManagementAPIAuthConfig struct {
	Mode   string                  `yaml:"mode,omitempty"`
	Tokens []ManagementAPITokenRef `yaml:"tokens,omitempty"`
	Roles  map[string][]string     `yaml:"roles,omitempty"`
}

// ManagementAPITokenRef binds a bearer token from an environment variable to a role.
type ManagementAPITokenRef struct {
	Env  string `yaml:"env,omitempty"`
	Role string `yaml:"role,omitempty"`
}

// ManagementAPIRuntimeOptions carries CLI overrides for management listener startup.
type ManagementAPIRuntimeOptions struct {
	Port           int
	BindAddress    string
	RemoteExposure *bool
	AuthMode       string
}

// DefaultManagementAPIConfig returns safe local defaults for the management listener.
func DefaultManagementAPIConfig() ManagementAPIConfig {
	return ManagementAPIConfig{
		BindAddress:    "127.0.0.1",
		Port:           8080,
		RemoteExposure: false,
		Auth: ManagementAPIAuthConfig{
			Mode:  ManagementAuthModeDisabled,
			Roles: DefaultManagementAPIRoles(),
		},
	}
}

// DefaultManagementAPIRoles returns the built-in role-to-permission map.
//
// secret_view is admin-only (via ManagementPermWildcard). Viewer and operator
// may call config.read routes such as GET /config/router, but secret fields are
// redacted unless the principal has secret_view.
//
// Keep config/config.yaml global.services.management_api.auth.roles aligned with
// these defaults; reference-config tests enforce that contract.
func DefaultManagementAPIRoles() map[string][]string {
	return map[string][]string{
		"viewer": {
			"health.read",
			"ready.read",
			"docs.read",
			"metrics.read",
			"classify.invoke",
			"config.read",
			"replay.read",
			"data.read",
			"cache.read",
			"compression.read",
		},
		"operator": {
			"health.read",
			"ready.read",
			"docs.read",
			"metrics.read",
			"classify.invoke",
			"config.read",
			"config.write",
			"learning.ingest",
			"replay.read",
			"replay.detail",
			"data.read",
			"data.write",
			"cache.read",
			"cache.invalidate",
			"compression.read",
			"compression.preview",
		},
		// Wildcard includes secret_view so admin can read plaintext secrets.
		"admin": {ManagementPermWildcard},
	}
}

// ResolvedManagementAPI merges config defaults with runtime CLI overrides.
func (c ManagementAPIConfig) ResolvedManagementAPI(opts ManagementAPIRuntimeOptions) (ManagementAPIConfig, error) {
	resolved := c
	resolved.applyManagementAPIDefaults(opts)
	if err := resolved.validateManagementAPI(); err != nil {
		return ManagementAPIConfig{}, err
	}
	return resolved, nil
}

func (c *ManagementAPIConfig) applyManagementAPIDefaults(opts ManagementAPIRuntimeOptions) {
	defaults := DefaultManagementAPIConfig()
	if c.BindAddress == "" {
		c.BindAddress = defaults.BindAddress
	}
	if opts.BindAddress != "" {
		c.BindAddress = opts.BindAddress
	}
	if c.Port == 0 {
		c.Port = defaults.Port
	}
	if opts.Port != 0 {
		c.Port = opts.Port
	}
	if opts.RemoteExposure != nil {
		c.RemoteExposure = *opts.RemoteExposure
	}
	if opts.AuthMode != "" {
		c.Auth.Mode = opts.AuthMode
	}
	if c.Auth.Mode == "" {
		c.Auth.Mode = ManagementAuthModeDisabled
	}
	if len(c.Auth.Roles) == 0 {
		c.Auth.Roles = DefaultManagementAPIRoles()
	}
}

func (c ManagementAPIConfig) validateManagementAPI() error {
	validators := []func() error{
		c.validateBindAddress,
		c.validatePort,
		c.validateAuthMode,
		c.validateExposurePolicy,
		c.validateBindExposureConsistency,
	}
	for _, validate := range validators {
		if err := validate(); err != nil {
			return err
		}
	}
	return nil
}

func (c ManagementAPIConfig) validateAuthMode() error {
	if c.Auth.Mode != ManagementAuthModeDisabled && c.Auth.Mode != ManagementAuthModeBearer {
		return fmt.Errorf("management_api.auth.mode must be disabled or bearer")
	}
	return nil
}

func (c ManagementAPIConfig) validatePort() error {
	if c.Port < 1 || c.Port > 65535 {
		return fmt.Errorf("management_api.port must be between 1 and 65535")
	}
	return nil
}

func (c ManagementAPIConfig) ListenAddress() string {
	return net.JoinHostPort(c.BindAddress, fmt.Sprintf("%d", c.Port))
}

func (c ManagementAPIConfig) validateBindAddress() error {
	if strings.TrimSpace(c.BindAddress) == "" {
		return fmt.Errorf("management_api.bind_address must not be empty")
	}
	if ip := net.ParseIP(c.BindAddress); ip != nil {
		return nil
	}
	if c.BindAddress == "localhost" {
		return nil
	}
	return fmt.Errorf("management_api.bind_address %q must be an IP address or localhost", c.BindAddress)
}

func (c ManagementAPIConfig) validateExposurePolicy() error {
	if !c.RemoteExposure {
		return nil
	}
	if c.Auth.Mode == ManagementAuthModeDisabled {
		return fmt.Errorf("management_api.remote_exposure requires auth.mode bearer")
	}
	if len(c.Auth.Tokens) == 0 {
		return fmt.Errorf("management_api.remote_exposure requires at least one auth.tokens entry")
	}
	return nil
}

func (c ManagementAPIConfig) validateBindExposureConsistency() error {
	if c.RemoteExposure || !isWideManagementBindAddress(c.BindAddress) {
		return nil
	}
	if strings.TrimSpace(os.Getenv(ManagementInternalListenerEnv)) == "true" {
		return nil
	}
	return fmt.Errorf(
		"management_api.bind_address %q requires remote_exposure: true (or set %s=true for container-local listeners)",
		c.BindAddress,
		ManagementInternalListenerEnv,
	)
}

func isWideManagementBindAddress(bindAddress string) bool {
	switch strings.TrimSpace(bindAddress) {
	case "", "0.0.0.0", "::", "[::]":
		return true
	default:
		if ip := net.ParseIP(strings.TrimSpace(bindAddress)); ip != nil {
			return ip.IsUnspecified()
		}
		return false
	}
}

// ResolvedManagementTokens materializes bearer tokens from configured env refs.
func (c ManagementAPIAuthConfig) ResolvedManagementTokens() map[string]string {
	tokens := make(map[string]string)
	for _, ref := range c.Tokens {
		env := strings.TrimSpace(ref.Env)
		role := strings.TrimSpace(ref.Role)
		if env == "" || role == "" {
			continue
		}
		value := strings.TrimSpace(os.Getenv(env))
		if value == "" {
			continue
		}
		tokens[value] = role
	}
	return tokens
}
