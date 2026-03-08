package router

import (
	"log"
	"net/http"
	"path/filepath"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func registerCoreRoutes(mux *http.ServeMux, app *backendapp.App) {
	cfg := app.Config
	lifecycle := resolveConfigLifecycle(app)
	access := newRouteAccess(app)
	mux.HandleFunc("/healthz", handlers.HealthCheck)
	mux.HandleFunc("/api/settings", handlers.SettingsHandler(cfg))
	mux.HandleFunc("/api/auth/session", handlers.AuthSessionHandler(resolveAuth(app)))
	mux.HandleFunc("/api/auth/logout", handlers.AuthLogoutHandler(resolveAuth(app)))
	mux.Handle("/api/setup/state", access.viewer(handlers.SetupStateHandlerWithService(lifecycle)))
	mux.Handle("/api/setup/validate", access.editor(handlers.SetupValidateHandlerWithService(lifecycle)))
	mux.Handle("/api/setup/activate", access.operator(handlers.SetupActivateHandlerWithService(lifecycle, cfg.ReadonlyMode)))
}

func registerConfigRoutes(mux *http.ServeMux, app *backendapp.App) {
	cfg := app.Config
	lifecycle := resolveConfigLifecycle(app)
	access := newRouteAccess(app)
	mux.Handle("/api/router/config/revisions", access.viewer(handlers.ConfigRevisionsHandlerWithService(lifecycle)))
	mux.Handle("/api/router/config/revisions/current", access.viewer(handlers.CurrentConfigRevisionHandlerWithService(lifecycle)))
	mux.Handle("/api/router/config/revisions/draft", access.editor(handlers.SaveConfigRevisionDraftHandlerWithService(lifecycle, cfg.ReadonlyMode)))
	mux.Handle("/api/router/config/revisions/validate", access.editor(handlers.ValidateConfigRevisionHandlerWithService(lifecycle, cfg.ReadonlyMode)))
	mux.Handle("/api/router/config/revisions/activate", access.operator(handlers.ActivateConfigRevisionHandlerWithService(lifecycle, cfg.ReadonlyMode)))
	mux.Handle("/api/router/config/all", access.viewer(handlers.ConfigHandlerWithService(lifecycle)))
	mux.Handle("/api/router/config/yaml", access.viewer(handlers.ConfigYAMLHandlerWithService(lifecycle)))
	mux.Handle("/api/router/config/update", access.editor(handlers.UpdateConfigHandlerWithService(lifecycle, cfg.ReadonlyMode)))
	mux.Handle("/api/router/config/deploy/preview", access.editor(handlers.DeployPreviewHandlerWithService(lifecycle)))
	mux.Handle("/api/router/config/deploy", access.operator(handlers.DeployHandlerWithService(lifecycle, cfg.ReadonlyMode)))
	mux.Handle("/api/router/config/rollback", access.operator(handlers.RollbackHandlerWithService(lifecycle, cfg.ReadonlyMode)))
	mux.Handle("/api/router/config/versions", access.viewer(handlers.ConfigVersionsHandlerWithService(lifecycle)))
	mux.Handle("/api/router/config/defaults", access.viewer(handlers.RouterDefaultsHandlerWithService(lifecycle)))
	mux.Handle("/api/router/config/defaults/update", access.editor(handlers.UpdateRouterDefaultsHandlerWithService(lifecycle, cfg.ReadonlyMode)))
	log.Printf("Config API endpoints registered: /api/router/config/*")
	log.Printf("Config revision API endpoints registered: /api/router/config/revisions, /api/router/config/revisions/current, /api/router/config/revisions/draft, /api/router/config/revisions/validate, /api/router/config/revisions/activate")
	log.Printf("Router defaults API endpoints registered: /api/router/config/defaults, /api/router/config/defaults/update")
}

func resolveConfigLifecycle(app *backendapp.App) *configlifecycle.Service {
	if app != nil && app.ConfigLifecycle != nil {
		return app.ConfigLifecycle
	}
	if app == nil || app.Config == nil {
		return configlifecycle.New("", "")
	}
	return configlifecycle.New(app.Config.AbsConfigPath, app.Config.ConfigDir)
}

func registerToolsRoutes(mux *http.ServeMux, app *backendapp.App) {
	cfg := app.Config
	access := newRouteAccess(app)
	mux.Handle("/api/tools-db", access.viewer(handlers.ToolsDBHandler(resolveToolsDBPath(cfg))))
	mux.Handle("/api/tools/web-search", access.viewer(handlers.WebSearchHandler()))
	mux.Handle("/api/tools/open-web", access.viewer(handlers.OpenWebHandler()))
	mux.Handle("/api/tools/fetch-raw", access.viewer(handlers.FetchRawHandler()))
	log.Printf("Tools API endpoints registered: /api/tools-db, /api/tools/web-search, /api/tools/open-web, /api/tools/fetch-raw")
}

func registerDashboardUtilityRoutes(mux *http.ServeMux, app *backendapp.App) {
	cfg := app.Config
	access := newRouteAccess(app)
	mux.Handle("/api/status", access.viewer(handlers.StatusHandler(cfg.RouterAPIURL, cfg.ConfigDir)))
	mux.Handle("/api/logs", access.viewer(handlers.LogsHandler(cfg.RouterAPIURL)))
	mux.Handle("/api/topology/test-query", access.viewer(handlers.TopologyTestQueryHandler(cfg.AbsConfigPath, cfg.RouterAPIURL)))
	log.Printf("Status API endpoint registered: /api/status")
	log.Printf("Logs API endpoint registered: /api/logs")
	log.Printf("Topology Test Query API endpoint registered: /api/topology/test-query (Router API: %s)", cfg.RouterAPIURL)
}

func resolveToolsDBPath(cfg *config.Config) string {
	toolsDBPath := filepath.Join(cfg.ConfigDir, "config", "tools_db.json")
	parsedCfg, err := routerconfig.Parse(cfg.AbsConfigPath)
	if err != nil {
		log.Printf("Warning: failed to parse config for tools_db_path, use the default path %s: %v", toolsDBPath, err)
		return toolsDBPath
	}
	if parsedCfg.Tools.ToolsDBPath != "" {
		return parsedCfg.Tools.ToolsDBPath
	}
	return toolsDBPath
}
