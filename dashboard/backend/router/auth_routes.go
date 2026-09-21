package router

import (
	"context"
	"log"
	"net/http"
	"strings"

	auth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

type authRouteSpec struct {
	path    string
	methods []string
	// session routes need a live login but no permission; the rest are public.
	session bool
}

var dashboardAuthRouteSpecs = []authRouteSpec{
	{path: "/api/auth/login", methods: []string{http.MethodPost}},
	{path: "/api/auth/logout", methods: []string{http.MethodPost}},
	{path: "/api/auth/me", methods: []string{http.MethodGet}, session: true},
	{path: "/api/auth/bootstrap/can-register", methods: []string{http.MethodGet}},
	{path: "/api/auth/bootstrap/register", methods: []string{http.MethodPost}},
	{path: "/api/auth/invitations/{token}", methods: []string{http.MethodGet}},
	{path: "/api/auth/invitations/{token}/accept", methods: []string{http.MethodPost}},
}

const authUnavailableResponse = `{"error":"Service not available","message":"Authentication service is not configured"}`

func setupAuthRoutes(routes *auth.PolicyMux, cfg *config.Config, setupResolver *setupmode.Resolver) *auth.Service {
	store, err := auth.NewStore(cfg.AuthDBPath)
	if err != nil {
		log.Printf("failed to init auth store: %v", err)
		registerAuthUnavailableRoutes(routes)
		return nil
	}

	authSvc := auth.NewService(store, cfg.JWTSecret, cfg.JWTExpiryHours)
	authSvc.SetAllowOpenBootstrap(cfg.AllowOpenBootstrap)
	authSvc.SetAllowedOrigins(cfg.AllowedOrigins)
	// The bootstrap gate reads the resolver on every unauthenticated
	// can-register / register call, so setup mode tracks the config file.
	if setupResolver != nil {
		authSvc.SetSetupModeFunc(setupResolver.Active)
	} else {
		// Fail closed. Leaving setupModeFn unset keeps the endpoint shut.
		// Installing a method value on a nil resolver would panic instead.
		log.Printf("WARNING: setup-mode resolver unavailable; the open bootstrap endpoint is failing closed")
	}
	if err := authSvc.EnsureBootstrapAdmin(
		context.Background(),
		cfg.BootstrapAdminEmail,
		cfg.BootstrapAdminPassword,
		cfg.BootstrapAdminName,
	); err != nil {
		log.Printf("failed to ensure bootstrap admin: %v", err)
	}

	registerAuthProxyRoutes(routes, authSvc)
	auth.RegisterAdminRoutes(routes, authSvc)
	return authSvc
}

func registerAuthUnavailableRoutes(routes *auth.PolicyMux) {
	unavailable := func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, authUnavailableResponse, http.StatusServiceUnavailable)
	}
	for _, spec := range dashboardAuthRouteSpecs {
		registerAuthRoute(routes, spec, unavailable)
	}
}

func registerAuthProxyRoutes(routes *auth.PolicyMux, authSvc *auth.Service) {
	authRoutes := auth.AuthRoutes(authSvc)
	for _, spec := range dashboardAuthRouteSpecs {
		registerAuthRoute(routes, spec, func(w http.ResponseWriter, r *http.Request) {
			authRoutes.ServeHTTP(w, r)
		})
	}
}

// registerAuthRoute installs the exact path and its trailing-slash alias under
// one contract. The alias is forwarded to the inner auth mux under its
// canonical path; that mux keeps its own method checks.
func registerAuthRoute(routes *auth.PolicyMux, spec authRouteSpec, handler http.HandlerFunc) {
	canonical := func(w http.ResponseWriter, r *http.Request) {
		if len(r.URL.Path) > 1 && strings.HasSuffix(r.URL.Path, "/") {
			cloneReq := *r
			cloneURL := *r.URL
			cloneURL.Path = strings.TrimSuffix(cloneURL.Path, "/")
			cloneURL.RawPath = strings.TrimSuffix(cloneURL.RawPath, "/")
			cloneReq.URL = &cloneURL
			r = &cloneReq
		}
		handler(w, r)
	}
	contracts := make([]auth.RouteContract, 0, 2)
	for _, pattern := range []string{spec.path, spec.path + "/{$}"} {
		if spec.session {
			contracts = append(contracts, auth.SessionRoute(pattern, auth.SensitivitySensitive, auth.ResourceOwnerAuth, spec.methods...))
			continue
		}
		contracts = append(contracts, auth.PublicRoute(pattern, spec.methods...))
	}
	routes.HandleGroup(contracts, http.HandlerFunc(canonical))
}

func wrapWithAuth(routes *auth.PolicyMux, authSvc *auth.Service) http.Handler {
	if authSvc != nil {
		return withSRBenchResponsePolicy(auth.AuthenticateRequest(authSvc, routes)(routes))
	}
	// authSvc is nil only when the auth store failed to initialize. Fail
	// closed: deny every route that requires authentication rather than
	// serving the entire control plane (config deploy/rollback, admin user
	// management, MCP tooling, proxy) unauthenticated. Public routes and the
	// static frontend remain reachable so the dashboard can surface the
	// misconfiguration.
	log.Printf("WARNING: auth service unavailable; authenticated routes are failing closed (503). Check AuthDBPath/JWT configuration.")
	return withSRBenchResponsePolicy(auth.ServiceUnavailableGuard(routes)(routes))
}
