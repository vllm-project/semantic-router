package router

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// Setup configures all routes and returns the configured mux.
func Setup(cfg *config.Config) *http.ServeMux {
	mux := http.NewServeMux()
	authSvc := setupAuthRoutes(mux, cfg)

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
	registerEvaluationRoutes(mux, cfg)
	SetupMCP(mux, cfg, openClawHandler)
	registerMLPipelineRoutes(mux, cfg, wf)
	registerOpenClawRoutes(mux, cfg, openClawHandler)
	registerProxyRoutes(mux, cfg)

	// Static frontend must be registered last.
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))
	return wrapWithAuth(mux, authSvc)
}
