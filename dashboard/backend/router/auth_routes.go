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

type dashboardManagementIdentity interface {
	auth.DashboardSessionRetirer
	auth.InvitationAuthority
	auth.FirstAdminProvisioner
}

type unavailableFirstAdminProvisioner struct{}

func (unavailableFirstAdminProvisioner) ProvisionFirstAdmin(context.Context, auth.FirstAdminIdentity) error {
	return auth.ErrFirstAdminProvisioningUnavailable
}

var dashboardAuthRouteSpecs = []authRouteSpec{
	{path: "/api/auth/login", method: http.MethodPost},
	{path: "/api/auth/logout", method: http.MethodPost},
	{path: "/api/auth/me", method: http.MethodGet},
	{path: "/api/auth/bootstrap/can-register", method: http.MethodGet},
	{path: "/api/auth/bootstrap/register", method: http.MethodPost},
	{path: "/api/auth/invitations/info", method: http.MethodGet},
	{path: "/api/auth/invitations/accept", method: http.MethodPost},
}

const authUnavailableResponse = `{"error":"Service not available","message":"Authentication service is not configured"}`

func setupAuthRoutes(
	mux *http.ServeMux,
	cfg *config.Config,
	managementIdentity dashboardManagementIdentity,
) *auth.Service {
	store, err := auth.NewStore(cfg.AuthDBPath)
	if err != nil {
		log.Printf("failed to init auth store: %v", err)
		registerAuthUnavailableRoutes(mux)
		return nil
	}

	authSvc := auth.NewService(store, cfg.JWTSecret, cfg.JWTExpiryHours)
	authSvc.ConfigureDashboardSessionRetirer(managementIdentity)
	authSvc.SetAllowOpenBootstrap(cfg.AllowOpenBootstrap)
	if cfg.RouterBootstrapTokenFile != "" {
		var firstAdminProvisioner auth.FirstAdminProvisioner = unavailableFirstAdminProvisioner{}
		if managementIdentity != nil {
			firstAdminProvisioner = managementIdentity
		}
		authSvc.ConfigureFirstAdminProvisioner(firstAdminProvisioner)
	}
	authSvc.ConfigureInvitations(managementIdentity, auth.SMTPInvitationMailer{
		Host: cfg.SMTPHost, Port: cfg.SMTPPort, Username: cfg.SMTPUsername,
		Password: cfg.SMTPPassword, From: cfg.SMTPFrom,
	}, cfg.DashboardPublicURL, cfg.DashboardIssuer)
	if cfg.RouterBootstrapTokenFile == "" {
		if err := authSvc.EnsureBootstrapAdmin(
			context.Background(),
			cfg.BootstrapAdminEmail,
			cfg.BootstrapAdminPassword,
			cfg.BootstrapAdminName,
		); err != nil {
			log.Printf("failed to ensure bootstrap admin: %v", err)
		}
	} else if cfg.BootstrapAdminEmail != "" || cfg.BootstrapAdminPassword != "" {
		log.Printf("Dashboard admin environment bootstrap is disabled for Router-managed first installation")
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
	// authSvc is nil only when the auth store failed to initialize. Fail
	// closed: deny every route that requires authentication rather than
	// serving the entire control plane (config deploy/rollback, admin user
	// management, Agent tooling, proxy) unauthenticated. Public routes and the
	// static frontend remain reachable so the dashboard can surface the
	// misconfiguration.
	log.Printf("WARNING: auth service unavailable; authenticated routes are failing closed (503). Check AuthDBPath/JWT configuration.")
	wrappedMux.Handle("/", auth.ServiceUnavailableGuard()(mux))
	return wrappedMux
}
