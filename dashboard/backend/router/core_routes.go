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
	mux.HandleFunc("/healthz", handlers.HealthCheck)
	mux.HandleFunc("/api/settings", handlers.SettingsHandler(cfg))
	mux.HandleFunc("/api/setup/state", handlers.SetupStateHandlerWithService(lifecycle))
	mux.HandleFunc("/api/setup/validate", handlers.SetupValidateHandlerWithService(lifecycle))
	mux.HandleFunc("/api/setup/activate", handlers.SetupActivateHandlerWithService(lifecycle, cfg.ReadonlyMode))
}

func registerConfigRoutes(mux *http.ServeMux, app *backendapp.App) {
	cfg := app.Config
	lifecycle := resolveConfigLifecycle(app)
	mux.HandleFunc("/api/router/config/revisions", handlers.ConfigRevisionsHandlerWithService(lifecycle))
	mux.HandleFunc("/api/router/config/revisions/current", handlers.CurrentConfigRevisionHandlerWithService(lifecycle))
	mux.HandleFunc("/api/router/config/revisions/draft", handlers.SaveConfigRevisionDraftHandlerWithService(lifecycle, cfg.ReadonlyMode))
	mux.HandleFunc("/api/router/config/revisions/validate", handlers.ValidateConfigRevisionHandlerWithService(lifecycle, cfg.ReadonlyMode))
	mux.HandleFunc("/api/router/config/revisions/activate", handlers.ActivateConfigRevisionHandlerWithService(lifecycle, cfg.ReadonlyMode))
	mux.HandleFunc("/api/router/config/all", handlers.ConfigHandlerWithService(lifecycle))
	mux.HandleFunc("/api/router/config/yaml", handlers.ConfigYAMLHandlerWithService(lifecycle))
	mux.HandleFunc("/api/router/config/update", handlers.UpdateConfigHandlerWithService(lifecycle, cfg.ReadonlyMode))
	mux.HandleFunc("/api/router/config/deploy/preview", handlers.DeployPreviewHandlerWithService(lifecycle))
	mux.HandleFunc("/api/router/config/deploy", handlers.DeployHandlerWithService(lifecycle, cfg.ReadonlyMode))
	mux.HandleFunc("/api/router/config/rollback", handlers.RollbackHandlerWithService(lifecycle, cfg.ReadonlyMode))
	mux.HandleFunc("/api/router/config/versions", handlers.ConfigVersionsHandlerWithService(lifecycle))
	mux.HandleFunc("/api/router/config/defaults", handlers.RouterDefaultsHandlerWithService(lifecycle))
	mux.HandleFunc("/api/router/config/defaults/update", handlers.UpdateRouterDefaultsHandlerWithService(lifecycle, cfg.ReadonlyMode))
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

func registerToolsRoutes(mux *http.ServeMux, cfg *config.Config) {
	mux.HandleFunc("/api/tools-db", handlers.ToolsDBHandler(resolveToolsDBPath(cfg)))
	mux.HandleFunc("/api/tools/web-search", handlers.WebSearchHandler())
	mux.HandleFunc("/api/tools/open-web", handlers.OpenWebHandler())
	mux.HandleFunc("/api/tools/fetch-raw", handlers.FetchRawHandler())
	log.Printf("Tools API endpoints registered: /api/tools-db, /api/tools/web-search, /api/tools/open-web, /api/tools/fetch-raw")
}

func registerDashboardUtilityRoutes(mux *http.ServeMux, cfg *config.Config) {
	mux.HandleFunc("/api/status", handlers.StatusHandler(cfg.RouterAPIURL, cfg.ConfigDir))
	mux.HandleFunc("/api/logs", handlers.LogsHandler(cfg.RouterAPIURL))
	mux.HandleFunc("/api/topology/test-query", handlers.TopologyTestQueryHandler(cfg.AbsConfigPath, cfg.RouterAPIURL))
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
