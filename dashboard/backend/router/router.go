package router

import (
	"net/http"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

// Setup configures all routes and returns the configured mux.
func Setup(app *backendapp.App) *http.ServeMux {
	cfg := app.Config
	mux := http.NewServeMux()
	registerCoreRoutes(mux, app)
	registerConfigRoutes(mux, app)
	registerToolsRoutes(mux, cfg)
	registerDashboardUtilityRoutes(mux, cfg)
	registerEvaluationRoutes(mux, cfg)

	openClawHandler := buildOpenClawHandler(cfg)
	SetupMCP(mux, cfg, openClawHandler)
	registerMLPipelineRoutes(mux, cfg)
	registerOpenClawRoutes(mux, cfg, openClawHandler)

	proxies := registerIntegrationRoutes(mux, cfg)
	registerSmartAPIRoutes(mux, proxies)
	registerMetricsRoutes(mux, cfg)
	registerPrometheusRoutes(mux, cfg)
	registerJaegerRoutes(mux, cfg, proxies)
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))

	return mux
}
