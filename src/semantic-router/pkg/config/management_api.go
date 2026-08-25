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
	ManagementAuthModeRouter   = "router"

	ManagementPermWildcard = "*"
)

// ManagementAPIConfig configures the Router Management HTTP surface. Enabled
// controls versioned Management routes; the process listener remains an
// independent startup concern so operational probes can stay available.
type ManagementAPIConfig struct {
	Enabled        bool                    `yaml:"enabled"`
	BindAddress    string                  `yaml:"bind_address,omitempty"`
	Port           int                     `yaml:"port,omitempty"`
	RemoteExposure bool                    `yaml:"remote_exposure,omitempty"`
	Auth           ManagementAPIAuthConfig `yaml:"auth,omitempty"`
	TLS            ManagementAPITLSConfig  `yaml:"tls,omitempty"`
}

// ManagementAPIAuthConfig controls authentication for non-health management routes.
type ManagementAPIAuthConfig struct {
	Mode                          string                       `yaml:"mode,omitempty"`
	Tokens                        []ManagementAPITokenRef      `yaml:"tokens,omitempty"`
	Roles                         map[string][]string          `yaml:"roles,omitempty"`
	TokenSigningKeyringFile       string                       `yaml:"token_signing_keyring_file,omitempty"`
	TokenSigningKeyringEnv        string                       `yaml:"token_signing_keyring_env,omitempty"`
	ServiceAccountHMACKeyringFile string                       `yaml:"service_account_hmac_keyring_file,omitempty"`
	ServiceAccountHMACKeyringEnv  string                       `yaml:"service_account_hmac_keyring_env,omitempty"`
	InvitationHMACKeyringFile     string                       `yaml:"invitation_hmac_keyring_file,omitempty"`
	InvitationHMACKeyringEnv      string                       `yaml:"invitation_hmac_keyring_env,omitempty"`
	ResponseKEKKeyringFile        string                       `yaml:"response_kek_keyring_file,omitempty"`
	ResponseKEKKeyringEnv         string                       `yaml:"response_kek_keyring_env,omitempty"`
	Bootstrap                     ManagementAPIBootstrapConfig `yaml:"bootstrap,omitempty"`
	Recovery                      ManagementAPIRecoveryConfig  `yaml:"recovery,omitempty"`
}

// ManagementAPITokenRef binds a bearer token from an environment variable to a role.
type ManagementAPITokenRef struct {
	Env  string `yaml:"env,omitempty"`
	Role string `yaml:"role,omitempty"`
}

// ManagementAPIRuntimeOptions carries CLI overrides for management listener startup.
type ManagementAPIRuntimeOptions struct {
	DurableRouting bool
	Port           int
	BindAddress    string
	RemoteExposure *bool
	AuthMode       string
}

// DefaultManagementAPIConfig returns safe local defaults for the management listener.
func DefaultManagementAPIConfig() ManagementAPIConfig {
	config := ManagementAPIConfig{
		BindAddress:    "127.0.0.1",
		Port:           8080,
		RemoteExposure: false,
		Auth: ManagementAPIAuthConfig{
			Mode: ManagementAuthModeDisabled,
		},
	}
	config.applySecurityDefaults()
	return config
}

// DefaultManagementAPIRoles returns the built-in role-to-permission map.
//
// secret_view is admin-only (via ManagementPermWildcard). Viewer and operator
// may call config.read introspection routes, but secret fields are redacted
// unless the principal has secret_view.
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
	if err := resolved.validateManagementAPI(opts.DurableRouting); err != nil {
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
		if opts.DurableRouting && c.Enabled {
			c.Auth.Mode = ManagementAuthModeRouter
		} else {
			c.Auth.Mode = ManagementAuthModeDisabled
		}
	}
	if !opts.DurableRouting && len(c.Auth.Roles) == 0 {
		c.Auth.Roles = DefaultManagementAPIRoles()
	}
	c.applySecurityDefaults()
}

func (c ManagementAPIConfig) validateManagementAPI(durableRouting bool) error {
	validators := []func() error{
		c.validateBindAddress,
		c.validatePort,
		c.validateAuthMode,
		func() error { return c.validateAuthModeForAuthority(durableRouting) },
		c.validateExposurePolicy,
		c.validateBindExposureConsistency,
		func() error { return validateManagementBootstrapSecurity(c) },
	}
	for _, validate := range validators {
		if err := validate(); err != nil {
			return err
		}
	}
	return nil
}

func (c ManagementAPIConfig) validateAuthMode() error {
	if c.Auth.Mode != "" && c.Auth.Mode != ManagementAuthModeDisabled && c.Auth.Mode != ManagementAuthModeBearer &&
		c.Auth.Mode != ManagementAuthModeRouter {
		return fmt.Errorf("management_api.auth.mode must be disabled, bearer, or router")
	}
	return nil
}

func (c ManagementAPIConfig) validateAuthModeForAuthority(durableRouting bool) error {
	if durableRouting && c.Enabled {
		if c.Auth.Mode != ManagementAuthModeRouter {
			return fmt.Errorf("enabled durable Management API requires management_api.auth.mode router")
		}
		if len(c.Auth.Tokens) != 0 || len(c.Auth.Roles) != 0 {
			return fmt.Errorf("durable Management rejects static management_api.auth tokens and roles")
		}
		return nil
	}
	if c.Auth.Mode == ManagementAuthModeRouter && (!durableRouting || !c.Enabled) {
		return fmt.Errorf("management_api.auth.mode router requires enabled Management API with global.stores.management.postgres")
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
		return fmt.Errorf("management_api.remote_exposure requires auth.mode bearer or router")
	}
	if c.Auth.Mode == ManagementAuthModeBearer && len(c.Auth.Tokens) == 0 {
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
