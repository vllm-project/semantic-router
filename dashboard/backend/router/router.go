package router

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// Server bundles the dashboard mux with lifecycle hooks for durable stores.
type Server struct {
	Handler http.Handler
	Close   func() error
}

// Setup configures all routes and returns the dashboard server bundle.
func Setup(cfg *config.Config) *Server {
	mux := http.NewServeMux()
	managementSessions := setupManagementSessions(mux, cfg)
	authSvc := setupAuthRoutes(mux, cfg, managementSessions)

	wf, err := workflowstore.Open(cfg.WorkflowDBPath, workflowstore.Options{
		LegacyOpenClawDir: cfg.OpenClawDataDir,
	})
	if err != nil {
		log.Fatalf("workflow store: %v", err)
	}

	mux.HandleFunc("/api/workflows/health", handlers.WorkflowHealthHandler(wf))
	log.Printf("Workflow health API registered: /api/workflows/health")
	openClawHandler := newOpenClawHandler(cfg, wf)
	registerCoreRoutes(mux, cfg)
	registerEvaluationRoutes(mux, cfg, managementSessions)
	registerMLPipelineRoutes(mux, cfg, wf)
	registerOpenClawRoutes(mux, cfg, openClawHandler)
	registerProxyRoutes(mux, cfg, managementSessions)

	// Static frontend must be registered last.
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))
	return &Server{
		Handler: wrapWithAuth(mux, authSvc),
		Close:   wf.Close,
	}
}

func setupManagementSessions(mux *http.ServeMux, cfg *config.Config) routerauth.ManagementIdentityProvider {
	if cfg.DashboardIssuer == "" || cfg.DashboardSigningKeyFile == "" || cfg.DashboardKeyID == "" {
		log.Printf("Dashboard issuer disabled: signing configuration is incomplete")
		return nil
	}
	signer, err := routerauth.LoadEd25519AssertionSigner(cfg.DashboardSigningKeyFile, cfg.DashboardKeyID)
	if err != nil {
		log.Printf("Dashboard issuer disabled: %v", err)
		return nil
	}
	issuerURL, err := routerauth.CanonicalIssuerURL(cfg.DashboardIssuer)
	if err != nil {
		log.Printf("Dashboard issuer disabled: %v", err)
		return nil
	}
	routerauth.RegisterIssuerDiscovery(mux, issuerURL, signer)
	if cfg.DashboardIssuerID == "" {
		log.Printf("Router Management browser sessions disabled: Dashboard issuer ID is not configured")
		return nil
	}
	provider, err := routerauth.NewManagementSessionProvider(routerauth.ManagementSessionOptions{
		RouterURL: cfg.RouterAPIURL, IssuerURL: issuerURL,
		IssuerID: cfg.DashboardIssuerID, Signer: signer,
		BootstrapTokenFile: cfg.RouterBootstrapTokenFile,
	})
	if err != nil {
		log.Printf("Router Management browser sessions disabled: %v", err)
		return nil
	}
	return provider
}
