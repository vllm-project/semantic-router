package router

import (
	"context"
	"log"
	"net/http"
	"os"
	"path/filepath"

	auth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

type coreRouteOptions struct {
	recipeStore              *recipe.Store
	modelVerificationAuditor handlers.ModelVerificationAuditor
}

type configRouteOptions struct {
	credentialStore          *recipe.Store
	modelVerificationAuditor handlers.ModelVerificationAuditor
}

// setupResolver is required, not an option, because the setup routes have no
// fallback source of truth for setup mode.
func registerCoreRoutes(routes *auth.PolicyMux, cfg *config.Config, setupResolver *setupmode.Resolver, routeOptions ...coreRouteOptions) {
	options := coreRouteOptions{}
	if len(routeOptions) > 0 {
		options = routeOptions[0]
	}
	store := selectedRecipeStore(cfg, []*recipe.Store{options.recipeStore})
	registerHealthAndSetupRoutes(routes, cfg, setupResolver)
	registerConfigRoutes(routes, cfg, configRouteOptions{
		credentialStore:          store,
		modelVerificationAuditor: options.modelVerificationAuditor,
	})
	registerToolRoutes(routes, cfg)
	registerStatusRoutes(routes, cfg, store)
	registerTopologyRoutes(routes, cfg, store)
	registerRecipeRoutes(routes, cfg, store)
	registerSecurityPolicyRoutes(routes, cfg)
}

func registerRecipeRoutes(routes *auth.PolicyMux, cfg *config.Config, stores ...*recipe.Store) {
	recipeDir := dashboardActiveRecipeDirectory(cfg)
	store := selectedRecipeStore(cfg, stores)
	service := recipe.NewService(recipe.Options{
		Directory:         recipeDir,
		Store:             store,
		RuntimeConfigPath: cfg.AbsConfigPath,
		RouterAPIURL:      cfg.RouterAPIURL,
	})
	activator := handlers.NewRecipeActivator(handlers.RecipeActivatorOptions{
		Store:        store,
		ConfigPath:   cfg.AbsConfigPath,
		ConfigDir:    cfg.ConfigDir,
		RouterAPIURL: cfg.RouterAPIURL,
	})
	recoverRecipeActivationOnStartup(cfg, activator.Recover)
	handler := handlers.NewRecipeHandler(service, handlers.WithRecipePackageCapabilities(store, activator, handlers.RecipePackageCapabilities{
		ServerReadonly:        cfg.ReadonlyMode,
		RuntimeConfigWritable: cfg.RuntimeConfigWritable,
		RecipeStoreWritable:   cfg.RecipeStoreWritable,
	}))
	routes.HandleFunc(auth.ProtectedRoute("/api/recipe", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handler.Descriptor)
	routes.HandleFunc(auth.ProtectedRoute("/api/recipe/probes", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handler.Probes)
	routes.HandleFunc(
		auth.Route(
			"/api/recipe/probes/",
			auth.ReadPolicy(http.MethodGet, auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig),
			auth.MutationPolicy(http.MethodPost, auth.PermTopologyRead, "recipe.probe", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 2<<20),
		),
		handler.ProbeAction,
	)
	routes.HandleFunc(auth.ProtectedRoute("/api/recipe/packages", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handler.Packages)
	routes.HandleFunc(auth.ProtectedRoute("/api/recipe/packages/", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handler.Packages)
	registerRecipeMutationRoutes(routes, handler)
	log.Printf("Active Recipe API endpoints registered: /api/recipe, /api/recipe/probes/*, /api/recipe/packages, /api/recipe/import, /api/recipe/activate/preview, /api/recipe/activate, /api/recipe/deactivate/preview, /api/recipe/deactivate")
}

func registerRecipeMutationRoutes(routes *auth.PolicyMux, handler *handlers.RecipeHandler) {
	const maxRecipePackageBytes int64 = 32 << 20
	for _, route := range []struct {
		pattern string
		action  string
		handler http.HandlerFunc
	}{
		{pattern: "/api/recipe/import", action: "recipe.import", handler: handler.ImportPackage},
		{pattern: "/api/recipe/import/", action: "recipe.import", handler: handler.ImportPackage},
		{pattern: "/api/recipe/activate", action: "recipe.activate", handler: handler.ActivatePackage},
		{pattern: "/api/recipe/activate/", action: "recipe.activate", handler: handler.ActivatePackage},
		{pattern: "/api/recipe/activate/preview", action: "recipe.activate.preview", handler: handler.PreviewPackageActivation},
		{pattern: "/api/recipe/activate/preview/", action: "recipe.activate.preview", handler: handler.PreviewPackageActivation},
		{pattern: "/api/recipe/deactivate", action: "recipe.deactivate", handler: handler.DeactivatePackage},
		{pattern: "/api/recipe/deactivate/", action: "recipe.deactivate", handler: handler.DeactivatePackage},
		{pattern: "/api/recipe/deactivate/preview", action: "recipe.deactivate.preview", handler: handler.PreviewPackageDeactivation},
		{pattern: "/api/recipe/deactivate/preview/", action: "recipe.deactivate.preview", handler: handler.PreviewPackageDeactivation},
	} {
		routes.HandleFunc(
			auth.ProtectedMutationRoute(
				route.pattern,
				auth.PermConfigDeploy,
				route.action,
				auth.SensitivitySensitive,
				auth.ResourceOwnerConfig,
				maxRecipePackageBytes,
				http.MethodPost,
			),
			route.handler,
		)
	}
}

func recoverRecipeActivationOnStartup(cfg *config.Config, recover func(context.Context) error) {
	if cfg.ReadonlyMode || !cfg.RuntimeConfigWritable {
		log.Printf("Active Recipe recovery skipped: runtime configuration mutation is disabled")
		return
	}
	if err := recover(context.Background()); err != nil {
		log.Printf("Active Recipe recovery remains incomplete: %v", err)
	}
}

func registerSecurityPolicyRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	handlers.SetSecurityPolicyConfigPaths(cfg.AbsConfigPath, cfg.ConfigDir)
	routes.HandleFunc(auth.Route(
		"/api/security/policy",
		auth.ReadPolicy(http.MethodGet, auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig),
		auth.MutationPolicy(http.MethodPut, auth.PermSecurityManage, "security.policy.update", auth.SensitivitySecret, auth.ResourceOwnerConfig, 4<<20),
	), func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			handlers.HandleGetSecurityPolicy(w, r)
		case http.MethodPut:
			if runtimeConfigReadonly {
				http.Error(w, "Dashboard is in read-only mode", http.StatusForbidden)
				return
			}
			handlers.HandleUpdateSecurityPolicy(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
	routes.HandleFunc(auth.ProtectedMutationRoute(
		"/api/security/policy/preview",
		auth.PermSecurityManage,
		"security.policy.preview",
		auth.SensitivitySecret,
		auth.ResourceOwnerConfig,
		4<<20,
		http.MethodPost,
	), func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			if runtimeConfigReadonly {
				http.Error(w, "Dashboard runtime configuration is read-only", http.StatusForbidden)
				return
			}
			handlers.HandlePreviewSecurityFragment(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
	log.Printf("Security Policy API endpoints registered: /api/security/policy, /api/security/policy/preview")
}

func registerHealthAndSetupRoutes(routes *auth.PolicyMux, cfg *config.Config, setupResolver *setupmode.Resolver) {
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	routes.HandleFunc(auth.PublicRoute("/healthz", http.MethodGet), handlers.HealthCheck)
	routes.HandleFunc(auth.ProtectedRoute("/api/settings", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handlers.SettingsHandler(cfg, setupResolver))
	routes.HandleFunc(auth.PublicRoute("/api/setup/state", http.MethodGet), handlers.SetupStateHandler(cfg.AbsConfigPath, setupResolver))
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/setup/import-remote", auth.PermConfigWrite, "setup.import_remote", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 64<<10, http.MethodPost),
		handlers.SetupImportRemoteHandler(cfg.AbsConfigPath, setupResolver),
	)
	routes.HandleFunc(
		auth.ProtectedBoundedRoute("/api/setup/validate", auth.PermConfigWrite, auth.SensitivitySensitive, auth.ResourceOwnerConfig, 16<<20, http.MethodPost),
		handlers.SetupValidateHandler(cfg.AbsConfigPath, setupResolver),
	)
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/setup/activate", auth.PermConfigDeploy, "setup.activate", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost),
		handlers.SetupActivateHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir, setupResolver),
	)
	routes.HandleFunc(auth.ProtectedRoute("/api/setup/presets", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handlers.PresetsHandler())
	routes.HandleFunc(auth.ProtectedBoundedRoute("/api/setup/presets/delta", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, 2<<20, http.MethodPost), handlers.PresetDeltaHandler())
}

func registerConfigRoutes(routes *auth.PolicyMux, cfg *config.Config, routeOptions ...configRouteOptions) {
	options := configRouteOptions{}
	if len(routeOptions) > 0 {
		options = routeOptions[0]
	}
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	routes.HandleFunc(auth.ProtectedRoute("/api/models/catalog", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handlers.ModelCatalogHandler(handlers.NewCLIModelCatalogSource(cfg.PythonPath)))
	routes.HandleFunc(
		auth.ProtectedDelegatedAuditRoute("/api/models/verify", auth.PermEvalRun, "model.inference_verify", auth.SensitivitySensitive, auth.ResourceOwnerInference, 2<<20, http.MethodPost),
		handlers.ModelVerificationHandler(cfg.AbsConfigPath, options.modelVerificationAuditor),
	)
	registerConfigReadRoutes(routes, cfg)
	registerConfigMutationRoutes(routes, cfg, runtimeConfigReadonly)
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/router/config/deploy/preview", auth.PermConfigDeploy, "config.deploy_preview", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 16<<20, http.MethodPost), handlers.DeployPreviewHandler(cfg.AbsConfigPath))
	routes.HandleFunc(auth.ProtectedRoute("/api/router/config/versions", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handlers.ConfigVersionsHandler(cfg.AbsConfigPath))
	routes.HandleFunc(auth.ProtectedRoute("/api/router/config/deployments", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handlers.ConfigDeploymentsHandler())
	routes.HandleFunc(auth.ProtectedRoute("/api/router/config/deployments/", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handlers.ConfigDeploymentDetailHandler())
	routes.HandleFunc(auth.ProtectedRoute("/api/router/config/active-projection", auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig, http.MethodGet), handlers.ActiveConfigProjectionHandler())
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/router/config/nl/verify", auth.PermConfigWrite, "config.nl.verify", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 2<<20, http.MethodPost), handlers.BuilderNLVerifyHandler(cfg.AbsConfigPath, cfg.EnvoyURL))
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/router/config/nl/generate/stream", auth.PermConfigWrite, "config.nl.generate_stream", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 2<<20, http.MethodPost), handlers.BuilderNLGenerateStreamHandler(cfg.AbsConfigPath, cfg.EnvoyURL))
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/router/config/nl/generate", auth.PermConfigWrite, "config.nl.generate", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 2<<20, http.MethodPost), handlers.BuilderNLGenerateHandler(cfg.AbsConfigPath, cfg.EnvoyURL))
	log.Printf("Config API endpoints registered: /api/models/catalog, /api/models/verify, /api/router/config/all, /api/router/config/yaml, /api/router/config/update, /api/router/config/nl/verify, /api/router/config/nl/generate/stream, /api/router/config/nl/generate, /api/router/config/deploy, /api/router/config/deploy/preview, /api/router/config/rollback, /api/router/config/versions, /api/router/config/deployments, /api/router/config/active-projection")

	routes.HandleFunc(auth.ProtectedRoute("/api/router/config/global", auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig, http.MethodGet), handlers.RouterDefaultsHandler(cfg.AbsConfigPath))
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/router/config/global/update", auth.PermConfigWrite, "config.global.update", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost, http.MethodPut), handlers.UpdateRouterDefaultsHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	routes.HandleFunc(auth.ProtectedRoute("/api/router/config/global/raw", auth.PermConfigRead, auth.SensitivitySecret, auth.ResourceOwnerConfig, http.MethodGet), handlers.GlobalConfigYAMLHandler(cfg.AbsConfigPath))
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/router/config/global/raw/update", auth.PermConfigWrite, "config.global_raw.update", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost, http.MethodPut), handlers.UpdateGlobalConfigYAMLHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	routes.HandleFunc(auth.ProtectedRoute("/api/router/config/defaults", auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig, http.MethodGet), handlers.RouterDefaultsHandler(cfg.AbsConfigPath))
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/router/config/defaults/update", auth.PermConfigWrite, "config.defaults.update", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost, http.MethodPut), handlers.UpdateRouterDefaultsHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	store := selectedRecipeStore(cfg, []*recipe.Store{options.credentialStore})
	classifierHandler := handlers.RouterClassifierProxyHandler(cfg.RouterAPIURL, cfg.ReadonlyMode, store)
	kbPolicies := auth.Route(
		"/api/router/config/kbs",
		auth.ReadPolicy(http.MethodGet, auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig),
		auth.MutationPolicy(http.MethodPost, auth.PermConfigWrite, "config.kb.create", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20),
	)
	routes.HandleFunc(kbPolicies, classifierHandler)
	kbItemPolicies := auth.Route(
		"/api/router/config/kbs/",
		auth.ReadPolicy(http.MethodGet, auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig),
		auth.MutationPolicy(http.MethodPut, auth.PermConfigWrite, "config.kb.update", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20),
		auth.MutationPolicy(http.MethodDelete, auth.PermConfigWrite, "config.kb.delete", auth.SensitivitySecret, auth.ResourceOwnerConfig, auth.NoBodyLimit),
	)
	routes.HandleFunc(kbItemPolicies, classifierHandler)
	log.Printf("Global config API endpoints registered: /api/router/config/global, /api/router/config/global/update, /api/router/config/global/raw, /api/router/config/global/raw/update (legacy aliases: /api/router/config/defaults, /api/router/config/defaults/update)")
}

func registerConfigReadRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	for _, route := range []struct {
		pattern string
		handler http.HandlerFunc
	}{
		{pattern: "/api/router/config/all", handler: handlers.ConfigHandler(cfg.AbsConfigPath)},
		{pattern: "/api/router/config/yaml", handler: handlers.ConfigYAMLHandler(cfg.AbsConfigPath)},
	} {
		routes.HandleFunc(
			auth.ProtectedRoute(route.pattern, auth.PermConfigRead, auth.SensitivitySecret, auth.ResourceOwnerConfig, http.MethodGet),
			route.handler,
		)
	}
}

func registerConfigMutationRoutes(routes *auth.PolicyMux, cfg *config.Config, readonly bool) {
	const maxConfigBodyBytes int64 = 16 << 20
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/router/config/update", auth.PermConfigWrite, "config.update", auth.SensitivitySecret, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost, http.MethodPut),
		handlers.UpdateConfigHandler(cfg.AbsConfigPath, readonly, cfg.ConfigDir),
	)
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/router/config/deploy", auth.PermConfigDeploy, "config.deploy", auth.SensitivitySecret, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost),
		handlers.DeployHandler(cfg.AbsConfigPath, readonly, cfg.ConfigDir),
	)
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/router/config/rollback", auth.PermConfigDeploy, "config.rollback", auth.SensitivitySecret, auth.ResourceOwnerConfig, 64<<10, http.MethodPost),
		handlers.RollbackHandler(cfg.AbsConfigPath, readonly, cfg.ConfigDir),
	)
}

func registerToolRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	toolsDBPath := resolveToolsDBPath(cfg)
	routes.HandleFunc(auth.ProtectedRoute("/api/tools-db", auth.PermMcpRead, auth.SensitivitySensitive, auth.ResourceOwnerTools, http.MethodGet), handlers.ToolsDBHandler(toolsDBPath))
	log.Printf("Tools DB API endpoint registered: /api/tools-db")

	routes.HandleFunc(auth.ProtectedMutationRoute("/api/tools/web-search", auth.PermToolsUse, "tool.web_search", auth.SensitivitySensitive, auth.ResourceOwnerTools, 256<<10, http.MethodPost), handlers.WebSearchHandler())
	log.Printf("Web Search API endpoint registered: /api/tools/web-search")

	routes.HandleFunc(auth.ProtectedMutationRoute("/api/tools/open-web", auth.PermToolsUse, "tool.open_web", auth.SensitivitySensitive, auth.ResourceOwnerTools, 256<<10, http.MethodPost), handlers.OpenWebHandler())
	log.Printf("Open Web API endpoint registered: /api/tools/open-web")

	routes.HandleFunc(auth.ProtectedMutationRoute("/api/tools/weather", auth.PermToolsUse, "tool.weather", auth.SensitivityOperational, auth.ResourceOwnerTools, 256<<10, http.MethodPost), handlers.WeatherHandler())
	log.Printf("Weather API endpoint registered: /api/tools/weather")

	routes.HandleFunc(auth.ProtectedMutationRoute("/api/tools/fetch-raw", auth.PermToolsUse, "tool.fetch_raw", auth.SensitivitySensitive, auth.ResourceOwnerTools, 256<<10, http.MethodPost), handlers.FetchRawHandler())
	log.Printf("Fetch Raw API endpoint registered: /api/tools/fetch-raw")
}

func resolveToolsDBPath(cfg *config.Config) string {
	toolsDBPath := filepath.Join(cfg.ConfigDir, "config", "tools_db.json")
	toolSelection, err := routercontract.ReadToolSelection(cfg.AbsConfigPath)
	if err != nil {
		log.Printf("Warning: failed to parse config for tools_db_path, use the default path %s: %v", toolsDBPath, err)
		return toolsDBPath
	}
	if toolSelection.ToolsDBPath != "" {
		return toolSelection.ToolsDBPath
	}
	return toolsDBPath
}

func registerStatusRoutes(routes *auth.PolicyMux, cfg *config.Config, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	routes.HandleFunc(auth.ProtectedRoute("/api/status", auth.PermTopologyRead, auth.SensitivityOperational, auth.ResourceOwnerObservability, http.MethodGet), handlers.StatusHandler(cfg.RouterAPIURL, cfg.ConfigDir, store))
	log.Printf("Status API endpoint registered: /api/status")

	routes.HandleFunc(auth.ProtectedRoute("/api/logs", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), handlers.LogsHandler(cfg.RouterAPIURL))
	log.Printf("Logs API endpoint registered: /api/logs")
}

func registerTopologyRoutes(routes *auth.PolicyMux, cfg *config.Config, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/topology/test-query", auth.PermTopologyRead, "topology.test_query", auth.SensitivitySensitive, auth.ResourceOwnerInference, 2<<20, http.MethodPost), handlers.TopologyTestQueryHandler(cfg.AbsConfigPath, cfg.RouterAPIURL, store))
	log.Printf("Topology Test Query API endpoint registered: /api/topology/test-query (Router API: %s)", cfg.RouterAPIURL)
}

func registerEvaluationRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	if !cfg.EvaluationEnabled {
		log.Printf("Evaluation feature disabled")
		return
	}

	routes.HandleFunc(auth.ProtectedRoute("/api/evaluation/datasets", auth.PermEvalRead, auth.SensitivityOperational, auth.ResourceOwnerEvaluation, http.MethodGet), handlers.GetDatasetsHandler())
	log.Printf("Evaluation datasets endpoint registered: /api/evaluation/datasets")

	projectRoot := resolveEvaluationProjectRoot(cfg)
	log.Printf("Evaluation project root: %s", projectRoot)

	evalDB, err := evaluation.NewDB(cfg.EvaluationDBPath)
	if err != nil {
		log.Printf("Warning: failed to initialize evaluation database: %v (other evaluation endpoints disabled)", err)
		return
	}

	// Recover tasks that were running before a dashboard restart so UI state is consistent
	if err := evalDB.RecoverRunningTasks("Dashboard restarted; task interrupted"); err != nil {
		log.Printf("Warning: failed to recover running evaluation tasks: %v", err)
	}

	runner := evaluation.NewRunner(evaluation.RunnerConfig{
		DB:            evalDB,
		ProjectRoot:   projectRoot,
		PythonPath:    cfg.PythonPath,
		ResultsDir:    cfg.EvaluationResultsDir,
		MaxConcurrent: 10,
	})
	evalHandler := handlers.NewEvaluationHandler(evalDB, runner, cfg.ReadonlyMode, cfg.RouterAPIURL, cfg.EnvoyURL)

	routes.HandleFunc(auth.Route(
		"/api/evaluation/tasks",
		auth.ReadPolicy(http.MethodGet, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation),
		auth.MutationPolicy(http.MethodPost, auth.PermEvalWrite, "evaluation.task.create", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, 2<<20),
	), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		switch r.Method {
		case http.MethodGet:
			evalHandler.ListTasksHandler().ServeHTTP(w, r)
		case http.MethodPost:
			evalHandler.CreateTaskHandler().ServeHTTP(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
	routes.HandleFunc(auth.Route(
		"/api/evaluation/tasks/",
		auth.ReadPolicy(http.MethodGet, auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation),
		auth.MutationPolicy(http.MethodDelete, auth.PermEvalWrite, "evaluation.task.delete", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, auth.NoBodyLimit),
	), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		switch r.Method {
		case http.MethodGet:
			evalHandler.GetTaskHandler().ServeHTTP(w, r)
		case http.MethodDelete:
			evalHandler.DeleteTaskHandler().ServeHTTP(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/evaluation/run", auth.PermEvalRun, "evaluation.run", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, 2<<20, http.MethodPost), evalHandler.RunTaskHandler())
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/evaluation/cancel/", auth.PermEvalRun, "evaluation.cancel", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, auth.NoBodyLimit, http.MethodPost), evalHandler.CancelTaskHandler())
	routes.HandleFunc(auth.ProtectedRoute("/api/evaluation/stream/", auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, http.MethodGet), evalHandler.StreamProgressHandler())
	routes.HandleFunc(auth.ProtectedRoute("/api/evaluation/results/", auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, http.MethodGet), evalHandler.GetResultsHandler())
	routes.HandleFunc(auth.ProtectedRoute("/api/evaluation/export/", auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, http.MethodGet), evalHandler.ExportResultsHandler())
	routes.HandleFunc(auth.ProtectedRoute("/api/evaluation/history", auth.PermEvalRead, auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, http.MethodGet), evalHandler.GetHistoryHandler())
	log.Printf("Evaluation API endpoints registered: /api/evaluation/*")
}

func resolveEvaluationProjectRoot(cfg *config.Config) string {
	for _, candidate := range evaluationProjectRootCandidates(cfg) {
		if root := findEvaluationProjectRoot(candidate); root != "" {
			return root
		}
	}

	projectRoot := filepath.Dir(cfg.ConfigDir)
	if projectRoot != "" && projectRoot != "." {
		return projectRoot
	}

	if wd, err := os.Getwd(); err == nil {
		return wd
	}

	return projectRoot
}

func evaluationProjectRootCandidates(cfg *config.Config) []string {
	candidates := []string{
		cfg.ConfigDir,
		filepath.Dir(cfg.ConfigDir),
		cfg.AbsConfigPath,
	}

	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, wd)
	}

	return candidates
}

func findEvaluationProjectRoot(start string) string {
	if start == "" {
		return ""
	}

	info, err := os.Stat(start)
	if err != nil {
		return ""
	}

	dir := filepath.Clean(start)
	if !info.IsDir() {
		dir = filepath.Dir(dir)
	}

	for {
		if isEvaluationProjectRoot(dir) {
			return dir
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

func isEvaluationProjectRoot(dir string) bool {
	requiredPaths := []string{
		filepath.Join("src", "training", "model_eval", "mmlu_pro_vllm_eval.py"),
		filepath.Join("src", "training", "model_eval", "signal_eval.py"),
	}

	for _, relPath := range requiredPaths {
		if _, err := os.Stat(filepath.Join(dir, relPath)); err != nil {
			return false
		}
	}

	return true
}

func registerMLPipelineRoutes(routes *auth.PolicyMux, cfg *config.Config, wf *workflowstore.Store) {
	if !cfg.MLPipelineEnabled {
		log.Printf("ML Pipeline feature disabled")
		return
	}

	trainingDir := resolveMLTrainingDir(cfg)
	mlRunner, err := mlpipeline.NewRunner(mlpipeline.RunnerConfig{
		DataDir:      cfg.MLPipelineDataDir,
		TrainingDir:  trainingDir,
		PythonPath:   cfg.PythonPath,
		MLServiceURL: cfg.MLServiceURL,
		Workflow:     wf,
	})
	if err != nil {
		log.Fatalf("ML pipeline runner: %v", err)
	}
	if err := wf.RecoverInterruptedMLJobs("interrupted by dashboard restart"); err != nil {
		log.Printf("ML pipeline: recover running jobs: %v", err)
	}
	mlHandler := handlers.NewMLPipelineHandler(mlRunner)

	routes.HandleFunc(auth.ProtectedRoute("/api/ml-pipeline/jobs", auth.PermMlPipeline, auth.SensitivitySensitive, auth.ResourceOwnerML, http.MethodGet), mlHandler.ListJobsHandler())
	routes.HandleFunc(auth.ProtectedRoute("/api/ml-pipeline/jobs/", auth.PermMlPipeline, auth.SensitivitySensitive, auth.ResourceOwnerML, http.MethodGet), mlHandler.GetJobHandler())
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/ml-pipeline/benchmark", auth.PermMlPipeline, "ml.benchmark", auth.SensitivitySensitive, auth.ResourceOwnerML, 2<<20, http.MethodPost), mlHandler.RunBenchmarkHandler())
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/ml-pipeline/train", auth.PermMlPipeline, "ml.train", auth.SensitivitySensitive, auth.ResourceOwnerML, 2<<20, http.MethodPost), mlHandler.RunTrainHandler())
	routes.HandleFunc(auth.ProtectedMutationRoute("/api/ml-pipeline/config", auth.PermMlPipeline, "ml.config.generate", auth.SensitivitySensitive, auth.ResourceOwnerML, 2<<20, http.MethodPost), mlHandler.GenerateConfigHandler())
	routes.HandleFunc(auth.ProtectedRoute("/api/ml-pipeline/download/", auth.PermMlPipeline, auth.SensitivitySensitive, auth.ResourceOwnerML, http.MethodGet), mlHandler.DownloadOutputHandler())
	routes.HandleFunc(auth.ProtectedRoute("/api/ml-pipeline/stream/", auth.PermMlPipeline, auth.SensitivitySensitive, auth.ResourceOwnerML, http.MethodGet), mlHandler.StreamProgressHandler())
	log.Printf("ML Pipeline API endpoints registered: /api/ml-pipeline/*")

	if trainingDir != "" {
		log.Printf("ML Training scripts directory: %s", trainingDir)
		return
	}
	log.Printf("Warning: ML training scripts directory not configured (set ML_TRAINING_DIR)")
}

func resolveMLTrainingDir(cfg *config.Config) string {
	if cfg.MLTrainingDir != "" {
		return cfg.MLTrainingDir
	}

	projectRoot := filepath.Dir(cfg.ConfigDir)
	candidate := filepath.Join(projectRoot, "src", "training", "ml_model_selection")
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	return ""
}
