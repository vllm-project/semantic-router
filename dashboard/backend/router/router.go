package router

import (
	"net/http"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

// Setup configures all routes and returns the configured mux.
func Setup(app *backendapp.App) *http.ServeMux {
	cfg := app.Config
	access := newRouteAccess(app)
	mux := http.NewServeMux()
	registerCoreRoutes(mux, app)
	registerConfigRoutes(mux, app)
	registerToolsRoutes(mux, app)
	registerDashboardUtilityRoutes(mux, app)
	registerEvaluationRoutes(mux, app)

	openClawHandler := buildOpenClawHandler(cfg)
	SetupMCP(mux, app, openClawHandler)
	registerMLPipelineRoutes(mux, app)
	registerOpenClawRoutes(mux, app, openClawHandler)

	proxies := registerIntegrationRoutes(mux, app)
	registerSmartAPIRoutes(mux, access, proxies)
	registerMetricsRoutes(mux, access, cfg)
	registerPrometheusRoutes(mux, app, access)
	registerJaegerRoutes(mux, app, access, proxies)
	mux.Handle("/", handlers.StaticFileServer(cfg.StaticDir))

	return mux
}
