package router

import (
	"context"
	"log"
	"net/http"

	auth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

type authRouteSpec struct {
	path   string
	method string
}

var dashboardAuthRouteSpecs = []authRouteSpec{
	{path: "/api/auth/login", method: http.MethodPost},
	{path: "/api/auth/logout", method: http.MethodPost},
	{path: "/api/auth/me", method: http.MethodGet},
	{path: "/api/auth/bootstrap/can-register", method: http.MethodGet},
	{path: "/api/auth/bootstrap/register", method: http.MethodPost},
	{path: "/api/auth/invitations", method: "*"},
}

const authUnavailableResponse = `{"error":"Service not available","message":"Authentication service is not configured"}`

func authRoutePolicy(spec authRouteSpec) auth.RouteContract {
	if spec.path == "/api/auth/me" {
		return auth.ProtectedRoute(spec.path, auth.PermSessionRead, auth.SensitivitySensitive, auth.ResourceOwnerAuth, spec.method)
	}
	if spec.method == "*" {
		return auth.PublicRoute(spec.path+"/", http.MethodGet, http.MethodPost)
	}
	return auth.PublicRoute(spec.path, spec.method)
}

func setupAuthRoutes(mux routeRegistrar, cfg *config.Config, setupResolver *setupmode.Resolver) *auth.Service {
	store, err := auth.NewStore(cfg.AuthDBPath)
	if err != nil {
		log.Printf("failed to init auth store: %v", err)
		registerAuthUnavailableRoutes(mux)
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

	registerAuthProxyRoutes(mux, authSvc)
	auth.RegisterAdminRoutes(mux, authSvc)
	return authSvc
}

func registerAuthUnavailableRoutes(mux routeRegistrar) {
	for _, spec := range dashboardAuthRouteSpecs {
		if spec.method == "*" {
			registerRouteFunc(mux, authRoutePolicy(spec), func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, authUnavailableResponse, http.StatusServiceUnavailable)
			})
			continue
		}
		registerAuthMethodRoute(mux, spec, func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, authUnavailableResponse, http.StatusServiceUnavailable)
		})
	}
}

func registerAuthProxyRoutes(mux routeRegistrar, authSvc *auth.Service) {
	authRoutes := auth.AuthRoutes(authSvc)
	for _, spec := range dashboardAuthRouteSpecs {
		if spec.method == "*" {
			registerRouteFunc(mux, authRoutePolicy(spec), func(w http.ResponseWriter, r *http.Request) {
				authRoutes.ServeHTTP(w, r)
			})
			continue
		}
		path := spec.path
		registerAuthMethodRoute(mux, spec, func(w http.ResponseWriter, r *http.Request) {
			cloneReq := *r
			cloneURL := *r.URL
			cloneURL.Path = path
			cloneReq.URL = &cloneURL
			authRoutes.ServeHTTP(w, &cloneReq)
		})
	}
}

func registerAuthMethodRoute(
	mux routeRegistrar,
	spec authRouteSpec,
	handler http.HandlerFunc,
) {
	wrapped := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != spec.method {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		handler(w, r)
	}
	registerRouteFunc(mux, authRoutePolicy(spec), wrapped)
	alias := spec
	alias.path += "/"
	registerRouteFunc(mux, authRoutePolicy(alias), wrapped)
}

func wrapWithAuth(mux routeRegistrar, authSvc *auth.Service, resolver ...auth.RoutePolicyResolver) *http.ServeMux {
	wrappedMux := http.NewServeMux()
	if authSvc != nil {
		wrappedMux.Handle("/", withSRBenchResponsePolicy(auth.AuthenticateRequest(authSvc, resolver...)(mux)))
		return wrappedMux
	}
	// authSvc is nil only when the auth store failed to initialize. Fail
	// closed: deny every route that requires authentication rather than
	// serving the entire control plane (config deploy/rollback, admin user
	// management, MCP tooling, proxy) unauthenticated. Public routes and the
	// static frontend remain reachable so the dashboard can surface the
	// misconfiguration.
	log.Printf("WARNING: auth service unavailable; authenticated routes are failing closed (503). Check AuthDBPath/JWT configuration.")
	wrappedMux.Handle("/", withSRBenchResponsePolicy(auth.ServiceUnavailableGuard(resolver...)(mux)))
	return wrappedMux
}
