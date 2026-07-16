//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type managementPrincipal struct {
	Role        string
	Anonymous   bool
	AuthEnabled bool
}

type managementAuthPolicy struct {
	Mode   string
	Roles  map[string][]string
	Tokens map[string]string
}

func (s *ClassificationAPIServer) managementAuthPolicy() managementAuthPolicy {
	cfg := s.managementAPIConfig()
	roles := cfg.Auth.Roles
	if len(roles) == 0 {
		roles = config.DefaultManagementAPIRoles()
	}
	return managementAuthPolicy{
		Mode:   cfg.Auth.Mode,
		Roles:  roles,
		Tokens: cfg.Auth.ResolvedManagementTokens(),
	}
}

func (s *ClassificationAPIServer) managementAPIConfig() config.ManagementAPIConfig {
	if s == nil || s.config == nil {
		return config.DefaultManagementAPIConfig()
	}
	cfg := s.config.ManagementAPI
	if cfg.BindAddress == "" && cfg.Port == 0 && cfg.Auth.Mode == "" {
		return config.DefaultManagementAPIConfig()
	}
	if cfg.Auth.Mode == "" {
		cfg.Auth.Mode = config.ManagementAuthModeDisabled
	}
	if len(cfg.Auth.Roles) == 0 {
		cfg.Auth.Roles = config.DefaultManagementAPIRoles()
	}
	return cfg
}

func (policy managementAuthPolicy) authorize(route apiRoute, r *http.Request) (managementPrincipal, int, string) {
	if route.Permission == PermHealthRead {
		return managementPrincipal{Role: "anonymous", Anonymous: true}, 0, ""
	}

	switch policy.Mode {
	case "", config.ManagementAuthModeDisabled:
		return managementPrincipal{Role: "admin", AuthEnabled: false}, 0, ""
	case config.ManagementAuthModeBearer:
		return policy.authorizeBearer(route, r)
	default:
		return managementPrincipal{}, http.StatusInternalServerError, "INVALID_AUTH_MODE"
	}
}

func (policy managementAuthPolicy) authorizeBearer(route apiRoute, r *http.Request) (managementPrincipal, int, string) {
	if len(policy.Tokens) == 0 {
		return managementPrincipal{}, http.StatusUnauthorized, "MANAGEMENT_AUTH_NOT_CONFIGURED"
	}

	token := extractBearerToken(r)
	if token == "" {
		return managementPrincipal{AuthEnabled: true}, http.StatusUnauthorized, "UNAUTHORIZED"
	}

	role, ok := policy.Tokens[token]
	if !ok {
		return managementPrincipal{AuthEnabled: true}, http.StatusUnauthorized, "UNAUTHORIZED"
	}
	principal := managementPrincipal{Role: role, AuthEnabled: true}
	if !principal.hasPermission(route.Permission, policy.Roles) {
		return principal, http.StatusForbidden, "FORBIDDEN"
	}
	return principal, 0, ""
}

func extractBearerToken(r *http.Request) string {
	if r == nil {
		return ""
	}
	authHeader := strings.TrimSpace(r.Header.Get("Authorization"))
	if authHeader == "" {
		return ""
	}
	const prefix = "Bearer "
	if !strings.HasPrefix(authHeader, prefix) {
		return ""
	}
	return strings.TrimSpace(strings.TrimPrefix(authHeader, prefix))
}

func (p managementPrincipal) hasPermission(required RoutePermission, roles map[string][]string) bool {
	if required == "" {
		return true
	}
	permissions, ok := roles[p.Role]
	if !ok {
		return false
	}
	for _, permission := range permissions {
		if permission == config.ManagementPermWildcard || RoutePermission(permission) == required {
			return true
		}
	}
	return false
}
