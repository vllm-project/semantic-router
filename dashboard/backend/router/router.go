package router

import (
	"errors"
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// Server bundles the dashboard mux with lifecycle hooks for durable stores.
type Server struct {
	Handler    http.Handler
	auth       *auth.Service
	workflow   *workflowstore.Store
	projection *configprojection.Store
	closeOnce  sync.Once
	closeErr   error
}

// Close releases every durable store owned by the server. It is idempotent.
func (s *Server) Close() error {
	if s == nil {
		return nil
	}
	s.closeOnce.Do(func() {
		var authErr, workflowErr, projectionErr error
		if s.auth != nil {
			authErr = s.auth.Close()
		}
		if s.workflow != nil {
			workflowErr = s.workflow.Close()
		}
		if s.projection != nil {
			projectionErr = s.projection.Close()
		}
		s.closeErr = errors.Join(authErr, workflowErr, projectionErr)
	})
	return s.closeErr
}

// Setup configures all routes and returns the dashboard server bundle.
func Setup(cfg *config.Config) (*Server, error) {
	mux := http.NewServeMux()
	authSvc, err := setupAuthRoutes(mux, cfg)
	if err != nil {
		return nil, fmt.Errorf("setup authentication: %w", err)
	}

	wf, err := workflowstore.Open(cfg.WorkflowDBPath, workflowstore.Options{
		LegacyOpenClawDir: cfg.OpenClawDataDir,
	})
	if err != nil {
		closeErr := authSvc.Close()
		return nil, errors.Join(
			fmt.Errorf("open workflow store: %w", err),
			closeErr,
		)
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
		Handler:    wrapWithAuth(mux, authSvc),
		auth:       authSvc,
		workflow:   wf,
		projection: cp,
	}, nil
}
