package config

import (
	"fmt"
	"sort"
)

// MaskingIdentityScope is the one scope a masking identity namespace takes:
// the verified principal of a request, or the installation when it has none.
const MaskingIdentityScope = "principal_or_installation"

// MaskingIdentityConfig names the identity namespaces that masking plugins in
// registry mode allocate placeholders from. A namespace keeps one placeholder
// per value for good, across requests, sessions and restarts.
type MaskingIdentityConfig struct {
	Namespaces map[string]MaskingIdentityNamespace `json:"namespaces,omitempty" yaml:"namespaces,omitempty"`
}

// MaskingIdentityNamespace is one namespace: the secret its value digests are
// keyed with, the scope that separates its placeholders, and the registry
// that records which index each digest holds.
type MaskingIdentityNamespace struct {
	// KeyFile holds the 32-byte namespace secret. KeyEnv instead names an
	// environment variable that holds it as 64 hex digits.
	KeyFile string `json:"key_file,omitempty" yaml:"key_file,omitempty"`
	KeyEnv  string `json:"key_env,omitempty" yaml:"key_env,omitempty"`
	Scope   string `json:"scope" yaml:"scope" jsonschema:"enum=principal_or_installation"`
	// SessionScope separates placeholders per root session as well.
	SessionScope bool                  `json:"session_scope,omitempty" yaml:"session_scope,omitempty"`
	Registry     MaskingRegistryConfig `json:"registry" yaml:"registry"`
}

// MaskingRegistryConfig is where a namespace's allocations live: an
// append-only file that one Router process owns, or Postgres for several.
type MaskingRegistryConfig struct {
	Backend  string                      `json:"backend" yaml:"backend" jsonschema:"enum=file,enum=postgres"`
	Path     string                      `json:"path,omitempty" yaml:"path,omitempty"`
	Postgres *RouterReplayPostgresConfig `json:"postgres,omitempty" yaml:"postgres,omitempty"`
}

// MaskingNamespaceNames returns the configured namespace names in order.
func (c MaskingIdentityConfig) MaskingNamespaceNames() []string {
	names := make([]string, 0, len(c.Namespaces))
	for name := range c.Namespaces {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func validateMaskingIdentity(cfg *RouterConfig) error {
	for _, name := range cfg.MaskingIdentity.MaskingNamespaceNames() {
		ns := cfg.MaskingIdentity.Namespaces[name]
		at := "global.services.masking_identity.namespaces." + name
		if ns.Scope != MaskingIdentityScope {
			return fmt.Errorf("%s.scope must be %s", at, MaskingIdentityScope)
		}
		if (ns.KeyFile == "") == (ns.KeyEnv == "") {
			return fmt.Errorf("%s needs exactly one of key_file and key_env", at)
		}
		switch ns.Registry.Backend {
		case "file":
			if ns.Registry.Path == "" {
				return fmt.Errorf("%s.registry.path is required for the file backend", at)
			}
		case "postgres":
			if ns.Registry.Postgres == nil {
				return fmt.Errorf("%s.registry.postgres is required for the postgres backend", at)
			}
		default:
			return fmt.Errorf("%s.registry.backend must be file or postgres", at)
		}
	}
	return nil
}

func (c *RouterConfig) resolveMaskingIdentity(p *MaskingPluginConfig) error {
	if !p.RegistryMode() {
		return nil
	}
	if _, ok := c.MaskingIdentity.Namespaces[p.IdentityNamespace]; !ok {
		return fmt.Errorf("identity_namespace %q names no namespace of global.services.masking_identity", p.IdentityNamespace)
	}
	return nil
}
