package router

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
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
	authSvc := setupAuthRoutes(mux, cfg)

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
		handlers.BootstrapConfigProjectionFromFile(cfg.AbsConfigPath, cfg.ConfigDir)
	}

	mux.HandleFunc("/api/workflows/health", handlers.WorkflowHealthHandler(wf))
	log.Printf("Workflow health API registered: /api/workflows/health")

	openClawHandler := newOpenClawHandler(cfg, wf)

	registerCoreRoutes(mux, cfg)
	registerEvaluationRoutes(mux, cfg)
	SetupMCP(mux, cfg, wf, openClawHandler)
	registerMLPipelineRoutes(mux, cfg, wf)
	registerOpenClawRoutes(mux, cfg, openClawHandler)
	registerProxyRoutes(mux, cfg)

	// Static frontend must be registered last.
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))
	return &Server{
		Handler: wrapWithAuth(mux, authSvc),
		Close: func() error {
			if cp == nil {
				return nil
			}
			return cp.Close()
		},
	}
}
