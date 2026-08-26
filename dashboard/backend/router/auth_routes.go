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
	for _, spec := range dashboardAuthRouteSpecs {
		registerAuthMethodRoute(routes, spec.path, spec.method, func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, authUnavailableResponse, http.StatusServiceUnavailable)
		})
	}
}

func registerAuthProxyRoutes(routes *auth.PolicyMux, authSvc *auth.Service) {
	authRoutes := auth.AuthRoutes(authSvc)
	for _, spec := range dashboardAuthRouteSpecs {
		path := spec.path
		registerAuthMethodRoute(routes, path, spec.method, func(w http.ResponseWriter, r *http.Request) {
			cloneReq := *r
			cloneURL := *r.URL
			cloneURL.Path = path
			cloneReq.URL = &cloneURL
			authRoutes.ServeHTTP(w, &cloneReq)
		})
	}
}

func registerAuthMethodRoute(
	routes *auth.PolicyMux,
	path string,
	method string,
	handler http.HandlerFunc,
) {
	wrapped := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		handler(w, r)
	}
	contract := auth.PublicRoute(path, method)
	if path == "/api/auth/me" {
		contract = auth.ProtectedRoute(
			path,
			auth.PermSessionRead,
			auth.SensitivitySensitive,
			auth.ResourceOwnerAuth,
			method,
		)
	}
	routes.HandleFunc(contract, wrapped)
	slashContract := contract
	slashContract.Pattern = path + "/"
	routes.HandleFunc(slashContract, wrapped)
}

func wrapWithAuth(routes *auth.PolicyMux, authSvc *auth.Service) *http.ServeMux {
	wrappedMux := http.NewServeMux()
	if authSvc != nil {
		wrappedMux.Handle("/", auth.AuthenticateRequest(authSvc, routes)(routes))
		return wrappedMux
	}
	// authSvc is nil only when the auth store failed to initialize. Fail
	// closed: deny every route that requires authentication rather than
	// serving the entire control plane (config deploy/rollback, admin user
	// management, MCP tooling, proxy) unauthenticated. Public routes and the
	// static frontend remain reachable so the dashboard can surface the
	// misconfiguration.
	log.Printf("WARNING: auth service unavailable; authenticated routes are failing closed (503). Check AuthDBPath/JWT configuration.")
	wrappedMux.Handle("/", auth.ServiceUnavailableGuard(routes)(routes))
	return wrappedMux
}
