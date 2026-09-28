package router

import (
	"context"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

type coreRouteOptions struct {
	recipeStore              *recipe.Store
	modelVerificationAuditor handlers.ModelVerificationAuditor
	statusHandler            http.HandlerFunc
}

type configRouteOptions struct {
	credentialStore          *recipe.Store
	modelVerificationAuditor handlers.ModelVerificationAuditor
}

// setupResolver is required, not an option, because the setup routes have no
// fallback source of truth for setup mode.
func registerCoreRoutes(mux routeRegistrar, cfg *config.Config, setupResolver *setupmode.Resolver, routeOptions ...coreRouteOptions) {
	options := coreRouteOptions{}
	if len(routeOptions) > 0 {
		options = routeOptions[0]
	}
	store := selectedRecipeStore(cfg, []*recipe.Store{options.recipeStore})
	registerHealthAndSetupRoutes(mux, cfg, setupResolver)
	registerConfigRoutes(mux, cfg, configRouteOptions{
		credentialStore:          store,
		modelVerificationAuditor: options.modelVerificationAuditor,
	})
	registerToolRoutes(mux, cfg)
	registerStatusRoutes(mux, cfg, options.statusHandler, store)
	registerTopologyRoutes(mux, cfg, store)
	registerRecipeRoutes(mux, cfg, store)
}

func registerRecipeRoutes(mux routeRegistrar, cfg *config.Config, stores ...*recipe.Store) {
	recipeDir := dashboardActiveRecipeDirectory(cfg)
	store := selectedRecipeStore(cfg, stores)
	service := recipe.NewService(recipe.Options{
		Directory:    recipeDir,
		Store:        store,
		RouterAPIURL: cfg.RouterAPIURL,
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
	registerRouteFunc(mux, auth.ProtectedRoute("/api/recipe", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handler.Descriptor)
	registerRouteFunc(mux, auth.ProtectedRoute("/api/recipe/probes", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handler.Probes)
	registerRouteFunc(mux, auth.Route("/api/recipe/probes/",
		auth.ReadPolicy(http.MethodGet, auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig),
		auth.MutationPolicy(http.MethodPost, auth.PermTopologyRead, "recipe.probe", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 2<<20),
	), handler.ProbeAction)
	for _, path := range []string{"/api/recipe/packages", "/api/recipe/packages/"} {
		registerRouteFunc(mux, auth.ProtectedRoute(path, auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handler.Packages)
	}
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/recipe/import", auth.PermConfigWrite, "recipe.import", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 32<<20, http.MethodPost), handler.ImportPackage)
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/recipe/import/", auth.PermConfigWrite, "recipe.import", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 32<<20, http.MethodPost), handler.ImportPackage)
	for _, route := range []struct {
		path, action string
		handler      http.HandlerFunc
	}{
		{"/api/recipe/activate", "recipe.activate", handler.ActivatePackage},
		{"/api/recipe/activate/", "recipe.activate", handler.ActivatePackage},
		{"/api/recipe/activate/preview", "recipe.activate.preview", handler.PreviewPackageActivation},
		{"/api/recipe/activate/preview/", "recipe.activate.preview", handler.PreviewPackageActivation},
		{"/api/recipe/deactivate", "recipe.deactivate", handler.DeactivatePackage},
		{"/api/recipe/deactivate/", "recipe.deactivate", handler.DeactivatePackage},
		{"/api/recipe/deactivate/preview", "recipe.deactivate.preview", handler.PreviewPackageDeactivation},
		{"/api/recipe/deactivate/preview/", "recipe.deactivate.preview", handler.PreviewPackageDeactivation},
	} {
		registerRouteFunc(mux, auth.ProtectedMutationRoute(route.path, auth.PermConfigDeploy, route.action, auth.SensitivitySensitive, auth.ResourceOwnerConfig, 32<<20, http.MethodPost), route.handler)
	}
	log.Printf("Active Recipe API endpoints registered: /api/recipe, /api/recipe/probes/*, /api/recipe/packages, /api/recipe/import, /api/recipe/activate/preview, /api/recipe/activate, /api/recipe/deactivate/preview, /api/recipe/deactivate")
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

func registerHealthAndSetupRoutes(mux routeRegistrar, cfg *config.Config, setupResolver *setupmode.Resolver) {
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	registerRouteFunc(mux, auth.PublicRoute("/healthz", http.MethodGet), handlers.HealthCheck)
	registerRouteFunc(mux, auth.ProtectedRoute("/api/settings", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handlers.SettingsHandler(cfg, setupResolver))
	registerRouteFunc(mux, auth.PublicRoute("/api/setup/state", http.MethodGet), handlers.SetupStateHandler(cfg.AbsConfigPath, setupResolver))
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/setup/import-remote", auth.PermConfigWrite, "setup.import_remote", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 64<<10, http.MethodPost), handlers.SetupImportRemoteHandler(cfg.AbsConfigPath, setupResolver))
	registerRouteFunc(mux, auth.ProtectedBoundedRoute("/api/setup/validate", auth.PermConfigWrite, auth.SensitivitySensitive, auth.ResourceOwnerConfig, 16<<20, http.MethodPost), handlers.SetupValidateHandler(cfg.AbsConfigPath, setupResolver))
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/setup/activate", auth.PermConfigDeploy, "setup.activate", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost), handlers.SetupActivateHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir, setupResolver))
	registerRouteFunc(mux, auth.ProtectedRoute("/api/setup/presets", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handlers.PresetsHandler())
	registerRouteFunc(mux, auth.ProtectedBoundedRoute("/api/setup/presets/delta", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, 2<<20, http.MethodPost), handlers.PresetDeltaHandler())
}

func registerConfigRoutes(mux routeRegistrar, cfg *config.Config, routeOptions ...configRouteOptions) {
	if err := handlers.RestrictExistingConfigSnapshots(cfg.ConfigDir); err != nil {
		log.Printf("Warning: could not restrict existing config snapshots: %v", err)
	}
	options := configRouteOptions{}
	if len(routeOptions) > 0 {
		options = routeOptions[0]
	}
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	store := selectedRecipeStore(cfg, []*recipe.Store{options.credentialStore})
	registerRouteFunc(mux, auth.ProtectedRoute("/api/models/catalog", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), handlers.ModelCatalogHandler(handlers.NewPackagedModelCatalogSource(cfg.PythonPath)))
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/models/discover", auth.PermConfigWrite, "model.discover", auth.SensitivitySensitive, auth.ResourceOwnerConfig, 2<<20, http.MethodPost), handlers.ModelDiscoveryHandler(nil))
	registerRouteFunc(mux, auth.ProtectedDelegatedAuditRoute("/api/models/verify", auth.PermEvalRun, "model.inference_verify", auth.SensitivitySensitive, auth.ResourceOwnerInference, 2<<20, http.MethodPost), handlers.ModelVerificationHandler(cfg.AbsConfigPath, options.modelVerificationAuditor))
	for _, route := range []struct {
		path    string
		handler http.HandlerFunc
	}{
		{"/api/router/config/all", handlers.ConfigHandler(cfg.AbsConfigPath)},
		{"/api/router/config/schema", handlers.ConfigSchemaHandler(cfg.RouterAPIURL, store)},
		{"/api/router/config/yaml", handlers.ConfigYAMLHandler(cfg.AbsConfigPath)},
		{"/api/router/config/versions", handlers.ConfigVersionsHandler(cfg.AbsConfigPath)},
		{"/api/router/config/deployments", handlers.ConfigDeploymentsHandler()},
		{"/api/router/config/deployments/", handlers.ConfigDeploymentDetailHandler()},
		{"/api/router/config/active-projection", handlers.ActiveConfigProjectionHandler()},
	} {
		registerRouteFunc(mux, auth.ProtectedRoute(route.path, auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig, http.MethodGet), route.handler)
	}
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/router/config/update", auth.PermConfigWrite, "config.update", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost, http.MethodPut), handlers.UpdateConfigHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	registerRouteFunc(mux, auth.ProtectedBoundedRoute("/api/router/config/deploy/preview", auth.PermConfigDeploy, auth.SensitivitySensitive, auth.ResourceOwnerConfig, 16<<20, http.MethodPost), handlers.DeployPreviewHandler(cfg.AbsConfigPath))
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/router/config/deploy", auth.PermConfigDeploy, "config.deploy", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost), handlers.DeployHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/router/config/rollback", auth.PermConfigDeploy, "config.rollback", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost), handlers.RollbackHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	log.Printf("Config API endpoints registered: /api/models/catalog, /api/models/discover, /api/models/verify, /api/router/config/all, /api/router/config/schema, /api/router/config/yaml, /api/router/config/update, /api/router/config/deploy, /api/router/config/deploy/preview, /api/router/config/rollback, /api/router/config/versions, /api/router/config/deployments, /api/router/config/active-projection")

	registerRouteFunc(mux, auth.ProtectedRoute("/api/router/config/global", auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig, http.MethodGet), handlers.RouterDefaultsHandler(cfg.AbsConfigPath))
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/router/config/global/update", auth.PermConfigWrite, "config.global.update", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost, http.MethodPut), handlers.UpdateRouterDefaultsHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	registerRouteFunc(mux, auth.ProtectedRoute("/api/router/config/global/raw", auth.PermConfigRead, auth.SensitivitySecret, auth.ResourceOwnerConfig, http.MethodGet), handlers.GlobalConfigYAMLHandler(cfg.AbsConfigPath))
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/router/config/global/raw/update", auth.PermConfigWrite, "config.global_raw.update", auth.SensitivitySecret, auth.ResourceOwnerConfig, 16<<20, http.MethodPost, http.MethodPut), handlers.UpdateGlobalConfigYAMLHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	registerKnowledgeBaseRoutes(mux, cfg, store)
	log.Printf("Global config API endpoints registered: /api/router/config/global, /api/router/config/global/update, /api/router/config/global/raw, /api/router/config/global/raw/update")
}

func registerToolRoutes(mux routeRegistrar, cfg *config.Config) {
	registerRouteFunc(mux, auth.ProtectedRoute("/api/tools-db", auth.PermToolsUse, auth.SensitivityOperational, auth.ResourceOwnerTools, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		// Configuration saves can change the selected database without restarting
		// Dashboard. Resolve the current canonical path for each refresh.
		handlers.ToolsDBHandler(resolveToolsDBPath(cfg))(w, r)
	})
	log.Printf("Tools DB API endpoint registered: /api/tools-db")

	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/tools/web-search", auth.PermToolsUse, "tools.web_search", auth.SensitivitySensitive, auth.ResourceOwnerTools, 2<<20, http.MethodPost), handlers.WebSearchHandler())
	log.Printf("Web Search API endpoint registered: /api/tools/web-search")

	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/tools/open-web", auth.PermToolsUse, "tools.open_web", auth.SensitivitySensitive, auth.ResourceOwnerTools, 2<<20, http.MethodPost), handlers.OpenWebHandler())
	log.Printf("Open Web API endpoint registered: /api/tools/open-web")

	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/tools/weather", auth.PermToolsUse, "tools.weather", auth.SensitivityOperational, auth.ResourceOwnerTools, 2<<20, http.MethodPost), handlers.WeatherHandler())
	log.Printf("Weather API endpoint registered: /api/tools/weather")

	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/tools/fetch-raw", auth.PermToolsUse, "tools.fetch_raw", auth.SensitivitySensitive, auth.ResourceOwnerTools, 2<<20, http.MethodPost), handlers.FetchRawHandler())
	log.Printf("Fetch Raw API endpoint registered: /api/tools/fetch-raw")
}

// defaultToolsDBPath mirrors the canonical Router default. Both configured and
// fallback paths use the explicit asset root, independently of the directory
// containing the runtime config or Dashboard's writable state.
const defaultToolsDBPath = "config/tools_db.json"

func resolveToolsDBPath(cfg *config.Config) string {
	projectRoot := cfg.ConfigBaseDir
	fallback := filepath.Join(projectRoot, defaultToolsDBPath)

	toolSelection, err := routercontract.ReadToolSelection(cfg.AbsConfigPath)
	if err != nil {
		log.Printf("Warning: failed to parse config for tools_db_path, use the default path %s: %v", fallback, err)
		return fallback
	}
	if toolSelection.ToolsDBPath == "" {
		return fallback
	}
	if filepath.IsAbs(toolSelection.ToolsDBPath) {
		return toolSelection.ToolsDBPath
	}
	return filepath.Join(projectRoot, toolSelection.ToolsDBPath)
}

func registerStatusRoutes(mux routeRegistrar, cfg *config.Config, statusHandler http.HandlerFunc, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	if statusHandler == nil {
		statusHandler = handlers.StatusHandler(cfg.RouterAPIURL, cfg.EnvoyURL, cfg.ConfigDir, store)
	}
	registerRouteFunc(mux, auth.PublicRoute("/api/status", http.MethodGet), statusHandler)
	log.Printf("Status API endpoint registered: /api/status")

	registerRouteFunc(mux, auth.ProtectedRoute("/api/logs", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet), handlers.LogsHandler(cfg.RouterAPIURL))
	log.Printf("Logs API endpoint registered: /api/logs")
}

func registerTopologyRoutes(mux routeRegistrar, cfg *config.Config, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/topology/test-query", auth.PermTopologyRead, "topology.test_query", auth.SensitivitySensitive, auth.ResourceOwnerEvaluation, 2<<20, http.MethodPost), handlers.TopologyTestQueryHandler(cfg.AbsConfigPath, cfg.RouterAPIURL, store))
	log.Printf("Topology Test Query API endpoint registered: /api/topology/test-query (Router API: %s)", cfg.RouterAPIURL)
}

func registerMLPipelineRoutes(mux routeRegistrar, cfg *config.Config, wf *workflowstore.Store) {
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

	registerRouteFunc(mux, auth.ProtectedRoute("/api/ml-pipeline/jobs", auth.PermMlPipeline, auth.SensitivitySensitive, auth.ResourceOwnerML, http.MethodGet), mlHandler.ListJobsHandler())
	registerRouteFunc(mux, auth.ProtectedRoute("/api/ml-pipeline/jobs/", auth.PermMlPipeline, auth.SensitivitySensitive, auth.ResourceOwnerML, http.MethodGet), mlHandler.GetJobHandler())
	registerRouteFunc(mux, auth.ProtectedStreamingMutationRoute("/api/ml-pipeline/benchmark", auth.PermMlPipeline, "ml.benchmark", auth.SensitivitySensitive, auth.ResourceOwnerML, handlers.MLBenchmarkUploadMaxBytes, http.MethodPost), mlHandler.RunBenchmarkHandler())
	registerRouteFunc(mux, auth.ProtectedStreamingMutationRoute("/api/ml-pipeline/train", auth.PermMlPipeline, "ml.train", auth.SensitivitySensitive, auth.ResourceOwnerML, handlers.MLTrainUploadMaxBytes, http.MethodPost), mlHandler.RunTrainHandler())
	registerRouteFunc(mux, auth.ProtectedMutationRoute("/api/ml-pipeline/config", auth.PermMlPipeline, "ml.config", auth.SensitivitySensitive, auth.ResourceOwnerML, 4<<20, http.MethodPost), mlHandler.GenerateConfigHandler())
	registerRouteFunc(mux, auth.ProtectedRoute("/api/ml-pipeline/download/", auth.PermMlPipeline, auth.SensitivitySecret, auth.ResourceOwnerML, http.MethodGet), mlHandler.DownloadOutputHandler())
	registerRouteFunc(mux, auth.ProtectedRoute("/api/ml-pipeline/stream/", auth.PermMlPipeline, auth.SensitivitySensitive, auth.ResourceOwnerML, http.MethodGet), mlHandler.StreamProgressHandler())
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
