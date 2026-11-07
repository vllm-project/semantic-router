package config

import (
	"fmt"
	"slices"
	"strings"
)

// DebugHeadersMode decides who may turn on the debug response headers with
// the x-vsr-debug request header.
type DebugHeadersMode string

const (
	// DebugHeadersNone ignores x-vsr-debug. An empty mode means the same. It
	// is not spelled "off", which YAML 1.1 parsers read as a boolean.
	DebugHeadersNone DebugHeadersMode = "none"
	// DebugHeadersAny honors x-vsr-debug from every client.
	DebugHeadersAny DebugHeadersMode = "any"
	// DebugHeadersAuthenticated honors x-vsr-debug only from the users and
	// groups in RouterHeadersConfig.DebugPrincipals, read from the identity
	// headers an authenticator sets (global.services.authz.identity).
	DebugHeadersAuthenticated DebugHeadersMode = "authenticated"
)

// SupportedDebugHeadersModes lists the values global.router.headers.debug accepts.
func SupportedDebugHeadersModes() []string {
	return []string{string(DebugHeadersNone), string(DebugHeadersAny), string(DebugHeadersAuthenticated)}
}

// RouterHeadersConfig governs the optional response headers the Router emits.
type RouterHeadersConfig struct {
	// Debug decides who may turn on the debug response headers. They name the
	// signal and guard rules a request matched, so they are off by default.
	Debug DebugHeadersMode `yaml:"debug,omitempty"`
	// DebugPrincipals are the callers that may turn them on when Debug is
	// authenticated.
	DebugPrincipals DebugPrincipals `yaml:"debug_principals,omitempty"`
}

// DebugPrincipals names authenticated users and groups.
type DebugPrincipals struct {
	Users  []string `yaml:"users,omitempty"`
	Groups []string `yaml:"groups,omitempty"`
}

func (p DebugPrincipals) empty() bool {
	return len(p.Users) == 0 && len(p.Groups) == 0
}

// AllowsDebugHeaders reports whether a caller may turn on the debug response
// headers. userID and groups are the caller's identity as the identity
// headers assert it; anonymous callers pass none.
func (h RouterHeadersConfig) AllowsDebugHeaders(userID string, groups []string) bool {
	switch h.Debug {
	case DebugHeadersAny:
		return true
	case DebugHeadersAuthenticated:
		if userID != "" && slices.Contains(h.DebugPrincipals.Users, userID) {
			return true
		}
		return slices.ContainsFunc(groups, func(group string) bool {
			return group != "" && slices.Contains(h.DebugPrincipals.Groups, group)
		})
	default:
		return false
	}
}

func validateRouterHeadersContracts(cfg *RouterConfig) error {
	headers := cfg.Headers
	switch headers.Debug {
	case "", DebugHeadersNone, DebugHeadersAny:
		if !headers.DebugPrincipals.empty() {
			return fmt.Errorf("global.router.headers.debug_principals applies only with debug: %s", DebugHeadersAuthenticated)
		}
	case DebugHeadersAuthenticated:
		if headers.DebugPrincipals.empty() {
			return fmt.Errorf("global.router.headers.debug: %s needs at least one user or group in debug_principals", DebugHeadersAuthenticated)
		}
		for _, name := range append(slices.Clone(headers.DebugPrincipals.Users), headers.DebugPrincipals.Groups...) {
			if name == "" || strings.TrimSpace(name) != name {
				return fmt.Errorf("global.router.headers.debug_principals: %q must be a name without surrounding spaces", name)
			}
		}
	default:
		return fmt.Errorf("global.router.headers.debug: %q is not supported, use one of: %s",
			headers.Debug, strings.Join(SupportedDebugHeadersModes(), ", "))
	}
	return nil
}
