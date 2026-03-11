package router

import (
	"context"
	"log"
	"net/http"

	auth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

type authRouteSpec struct {
	path   string
	method string
}

var dashboardAuthRouteSpecs = []authRouteSpec{
	{path: "/api/auth/login", method: http.MethodPost},
	{path: "/api/auth/me", method: http.MethodGet},
	{path: "/api/auth/bootstrap/can-register", method: http.MethodGet},
	{path: "/api/auth/bootstrap/register", method: http.MethodPost},
}

const authUnavailableResponse = `{"error":"Service not available","message":"Authentication service is not configured"}`

func setupAuthRoutes(mux *http.ServeMux, cfg *config.Config) *auth.Service {
	store, err := auth.NewStore(cfg.AuthDBPath)
	if err != nil {
		log.Printf("failed to init auth store: %v", err)
		registerAuthUnavailableRoutes(mux)
		return nil
	}

	authSvc := auth.NewService(store, cfg.JWTSecret, cfg.JWTExpiryHours)
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

func registerAuthUnavailableRoutes(mux *http.ServeMux) {
	for _, spec := range dashboardAuthRouteSpecs {
		registerAuthMethodRoute(mux, spec.path, spec.method, func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, authUnavailableResponse, http.StatusServiceUnavailable)
		})
	}
}

func registerAuthProxyRoutes(mux *http.ServeMux, authSvc *auth.Service) {
	authRoutes := auth.AuthRoutes(authSvc)
	for _, spec := range dashboardAuthRouteSpecs {
		path := spec.path
		registerAuthMethodRoute(mux, path, spec.method, func(w http.ResponseWriter, r *http.Request) {
			cloneReq := *r
			cloneURL := *r.URL
			cloneURL.Path = path
			cloneReq.URL = &cloneURL
			authRoutes.ServeHTTP(w, &cloneReq)
		})
	}
}

func registerAuthMethodRoute(
	mux *http.ServeMux,
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
	mux.HandleFunc(path, wrapped)
	mux.HandleFunc(path+"/", wrapped)
}

func wrapWithAuth(mux *http.ServeMux, authSvc *auth.Service) *http.ServeMux {
	wrappedMux := http.NewServeMux()
	if authSvc != nil {
		wrappedMux.Handle("/", auth.AuthenticateRequest(authSvc)(mux))
		return wrappedMux
	}
	wrappedMux.Handle("/", mux)
	return wrappedMux
}
