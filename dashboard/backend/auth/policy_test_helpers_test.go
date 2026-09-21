package auth

import (
	"net/http"
	"strings"
)

// resolverFunc adapts a function to RoutePolicyResolver for middleware tests
// that exercise credential and CSRF handling on arbitrary paths.
type resolverFunc func(method, path string) (RoutePolicy, RouteLookup)

func (f resolverFunc) LookupRoutePolicy(method, path string) (RoutePolicy, RouteLookup) {
	return f(method, path)
}

// protectedResolver admits every request under the listed permissions, except
// the login route, which stays public.
func protectedResolver(permissions ...string) RoutePolicyResolver {
	return resolverFunc(func(method, path string) (RoutePolicy, RouteLookup) {
		if path == "/api/auth/login" {
			return PublicPolicy(method), RouteFound
		}
		return RoutePolicy{
			Method:        method,
			Permissions:   permissions,
			AuditMode:     AuditNone,
			Sensitivity:   SensitivitySensitive,
			ResourceOwner: ResourceOwnerConfig,
		}, RouteFound
	})
}

// csrfTestResolver classifies the paths the CSRF suites probe the way the
// Dashboard registry does: observability proxies need logs.read, the admin
// API needs users.manage, login is public, everything else needs config.read.
func csrfTestResolver() RoutePolicyResolver {
	return resolverFunc(func(method, path string) (RoutePolicy, RouteLookup) {
		permission := PermConfigRead
		switch {
		case path == "/api/auth/login":
			return PublicPolicy(method), RouteFound
		case strings.HasPrefix(path, "/embedded/grafana/"), strings.HasPrefix(path, "/embedded/jaeger"), path == "/api/ds/query":
			permission = PermLogsRead
		case strings.HasPrefix(path, "/api/admin/"):
			permission = PermUsersManage
		}
		return RoutePolicy{
			Method:        method,
			Permissions:   []string{permission},
			AuditMode:     AuditNone,
			Sensitivity:   SensitivitySensitive,
			ResourceOwner: ResourceOwnerConfig,
		}, RouteFound
	})
}

// adminTestHandler binds the admin API to a fresh registry behind
// AuthenticateRequest, the way the Dashboard server wires it.
func adminTestHandler(svc *Service) http.Handler {
	routes := NewPolicyMux()
	RegisterAdminRoutes(routes, svc)
	return AuthenticateRequest(svc, routes)(routes)
}
