package router

import (
	"errors"
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	"github.com/vllm-project/semantic-router/dashboard/backend/statusstore"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// Server bundles the dashboard mux with lifecycle hooks for durable stores.
type Server struct {
	Handler http.Handler
	Close   func() error
	// routes is the sealed registry every request is authorized against.
	routes *auth.PolicyMux
}

// RouteContracts returns the registered route inventory for contract tests.
func (s *Server) RouteContracts() []auth.RouteContract {
	return s.routes.Contracts()
}

// Setup configures all routes and returns the dashboard server bundle.
//
// setupResolver is built by main, not here, so that the process has exactly one
// resolver and one cache over the config file.
func Setup(cfg *config.Config, setupResolver *setupmode.Resolver) *Server {
	routes := auth.NewPolicyMux()

	// The bootstrap gate consults the resolver on every unauthenticated
	// can-register / register call, so it must be wired before any request
	// arrives. Wiring it later compiles but panics at request time.
	authSvc := setupAuthRoutes(routes, cfg, setupResolver)

	wf, err := workflowstore.Open(cfg.WorkflowDBPath, workflowstore.Options{
		LegacyOpenClawDir: cfg.OpenClawDataDir,
	})
	if err != nil {
		log.Fatalf("workflow store: %v", err)
	}

	var cp *configprojection.Store
	if opened, openErr := configprojection.Open(cfg.ConfigProjectionDBPath); openErr != nil {
		log.Printf(
			"Warning: config projection store unavailable at %s: %v; deploy/update projection refresh and projection APIs will be degraded",
			cfg.ConfigProjectionDBPath,
			openErr,
		)
	} else {
		cp = opened
		handlers.SetConfigProjectionStore(cp)
	}

	routes.HandleFunc(
		auth.ProtectedRoute("/api/workflows/health", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerWorkflow, http.MethodGet),
		handlers.WorkflowHealthHandler(wf),
	)
	log.Printf("Workflow health API registered: /api/workflows/health")

	openClawHandler := newOpenClawHandler(cfg, wf)
	recipeStore := newDashboardRecipeStore(cfg)
	statusHistory, err := statusstore.Open(cfg.StatusDBPath)
	if err != nil {
		log.Printf("Warning: status history is unavailable: %v", err)
		statusHistory = nil
	}
	statusMonitor := handlers.NewStatusMonitor(cfg.RouterAPIURL, cfg.EnvoyURL, cfg.ConfigDir, statusHistory, recipeStore)
	statusMonitor.Start()

	registerCoreRoutes(routes, cfg, setupResolver, coreRouteOptions{
		recipeStore:              recipeStore,
		modelVerificationAuditor: authSvc,
		statusHandler:            statusMonitor.Handler(),
	})
	registerSRBenchRoutes(routes, cfg)
	SetupMCP(routes, cfg, wf, openClawHandler)
	registerMLPipelineRoutes(routes, cfg, wf)
	registerOpenClawRoutes(routes, cfg, openClawHandler)
	registerProxyRoutes(routes, cfg, authSvc, recipeStore)

	// Static frontend must be registered last.
	routes.HandleFallback("/", handlers.StaticFileServer(cfg.StaticDir))
	routes.Seal()
	return &Server{
		Handler: wrapWithAuth(routes, authSvc),
		routes:  routes,
		Close: func() error {
			var projectionClose error
			if cp != nil {
				projectionClose = cp.Close()
			}
			return errors.Join(statusMonitor.Close(), statusHistory.Close(), projectionClose, wf.Close())
		},
	}
}
