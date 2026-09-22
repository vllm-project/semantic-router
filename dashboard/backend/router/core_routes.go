package router

import (
	"context"
	"log"
	"net/http"
	"os"
	"path/filepath"

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
func registerCoreRoutes(mux *http.ServeMux, cfg *config.Config, setupResolver *setupmode.Resolver, routeOptions ...coreRouteOptions) {
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

func registerRecipeRoutes(mux *http.ServeMux, cfg *config.Config, stores ...*recipe.Store) {
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
	mux.HandleFunc("/api/recipe", handler.Descriptor)
	mux.HandleFunc("/api/recipe/probes", handler.Probes)
	mux.HandleFunc("/api/recipe/probes/", handler.ProbeAction)
	mux.HandleFunc("/api/recipe/packages", handler.Packages)
	mux.HandleFunc("/api/recipe/packages/", handler.Packages)
	mux.HandleFunc("/api/recipe/import", handler.ImportPackage)
	mux.HandleFunc("/api/recipe/import/", handler.ImportPackage)
	mux.HandleFunc("/api/recipe/activate", handler.ActivatePackage)
	mux.HandleFunc("/api/recipe/activate/", handler.ActivatePackage)
	mux.HandleFunc("/api/recipe/activate/preview", handler.PreviewPackageActivation)
	mux.HandleFunc("/api/recipe/activate/preview/", handler.PreviewPackageActivation)
	mux.HandleFunc("/api/recipe/deactivate", handler.DeactivatePackage)
	mux.HandleFunc("/api/recipe/deactivate/", handler.DeactivatePackage)
	mux.HandleFunc("/api/recipe/deactivate/preview", handler.PreviewPackageDeactivation)
	mux.HandleFunc("/api/recipe/deactivate/preview/", handler.PreviewPackageDeactivation)
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

func registerHealthAndSetupRoutes(mux *http.ServeMux, cfg *config.Config, setupResolver *setupmode.Resolver) {
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	mux.HandleFunc("/healthz", handlers.HealthCheck)
	mux.HandleFunc("/api/settings", handlers.SettingsHandler(cfg, setupResolver))
	mux.HandleFunc("/api/setup/state", handlers.SetupStateHandler(cfg.AbsConfigPath, setupResolver))
	mux.HandleFunc("/api/setup/import-remote", handlers.SetupImportRemoteHandler(cfg.AbsConfigPath, setupResolver))
	mux.HandleFunc("/api/setup/validate", handlers.SetupValidateHandler(cfg.AbsConfigPath, setupResolver))
	mux.HandleFunc("/api/setup/activate", handlers.SetupActivateHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir, setupResolver))
	mux.HandleFunc("/api/setup/presets", handlers.PresetsHandler())
	mux.HandleFunc("/api/setup/presets/delta", handlers.PresetDeltaHandler())
}

func registerConfigRoutes(mux *http.ServeMux, cfg *config.Config, routeOptions ...configRouteOptions) {
	if err := handlers.RestrictExistingConfigSnapshots(cfg.ConfigDir); err != nil {
		log.Printf("Warning: could not restrict existing config snapshots: %v", err)
	}
	options := configRouteOptions{}
	if len(routeOptions) > 0 {
		options = routeOptions[0]
	}
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	store := selectedRecipeStore(cfg, []*recipe.Store{options.credentialStore})
	mux.HandleFunc("/api/models/catalog", handlers.ModelCatalogHandler(handlers.NewPackagedModelCatalogSource(cfg.PythonPath)))
	mux.HandleFunc("/api/models/discover", handlers.ModelDiscoveryHandler(nil))
	mux.HandleFunc("/api/models/verify", handlers.ModelVerificationHandler(cfg.AbsConfigPath, options.modelVerificationAuditor))
	mux.HandleFunc("/api/router/config/all", handlers.ConfigHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/schema", handlers.ConfigSchemaHandler(cfg.RouterAPIURL, store))
	mux.HandleFunc("/api/router/config/yaml", handlers.ConfigYAMLHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/update", handlers.UpdateConfigHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/deploy/preview", handlers.DeployPreviewHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/deploy", handlers.DeployHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/rollback", handlers.RollbackHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/versions", handlers.ConfigVersionsHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/deployments", handlers.ConfigDeploymentsHandler())
	mux.HandleFunc("/api/router/config/deployments/", handlers.ConfigDeploymentDetailHandler())
	mux.HandleFunc("/api/router/config/active-projection", handlers.ActiveConfigProjectionHandler())
	log.Printf("Config API endpoints registered: /api/models/catalog, /api/models/discover, /api/models/verify, /api/router/config/all, /api/router/config/schema, /api/router/config/yaml, /api/router/config/update, /api/router/config/deploy, /api/router/config/deploy/preview, /api/router/config/rollback, /api/router/config/versions, /api/router/config/deployments, /api/router/config/active-projection")

	mux.HandleFunc("/api/router/config/global", handlers.RouterDefaultsHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/global/update", handlers.UpdateRouterDefaultsHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/global/raw", handlers.GlobalConfigYAMLHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/global/raw/update", handlers.UpdateGlobalConfigYAMLHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/api/v1/storage/knowledge-bases", handlers.RouterClassifierProxyHandler(cfg.RouterAPIURL, cfg.ReadonlyMode, store))
	mux.HandleFunc("/api/router/api/v1/storage/knowledge-bases/", handlers.RouterClassifierProxyHandler(cfg.RouterAPIURL, cfg.ReadonlyMode, store))
	log.Printf("Global config API endpoints registered: /api/router/config/global, /api/router/config/global/update, /api/router/config/global/raw, /api/router/config/global/raw/update")
}

func registerToolRoutes(mux *http.ServeMux, cfg *config.Config) {
	mux.HandleFunc("/api/tools-db", func(w http.ResponseWriter, r *http.Request) {
		// Configuration saves can change the selected database without restarting
		// Dashboard. Resolve the current canonical path for each refresh.
		handlers.ToolsDBHandler(resolveToolsDBPath(cfg))(w, r)
	})
	log.Printf("Tools DB API endpoint registered: /api/tools-db")

	mux.HandleFunc("/api/tools/web-search", handlers.WebSearchHandler())
	log.Printf("Web Search API endpoint registered: /api/tools/web-search")

	mux.HandleFunc("/api/tools/open-web", handlers.OpenWebHandler())
	log.Printf("Open Web API endpoint registered: /api/tools/open-web")

	mux.HandleFunc("/api/tools/weather", handlers.WeatherHandler())
	log.Printf("Weather API endpoint registered: /api/tools/weather")

	mux.HandleFunc("/api/tools/fetch-raw", handlers.FetchRawHandler())
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

func registerStatusRoutes(mux *http.ServeMux, cfg *config.Config, statusHandler http.HandlerFunc, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	if statusHandler == nil {
		statusHandler = handlers.StatusHandler(cfg.RouterAPIURL, cfg.EnvoyURL, cfg.ConfigDir, store)
	}
	mux.HandleFunc("/api/status", statusHandler)
	log.Printf("Status API endpoint registered: /api/status")

	mux.HandleFunc("/api/logs", handlers.LogsHandler(cfg.RouterAPIURL))
	log.Printf("Logs API endpoint registered: /api/logs")
}

func registerTopologyRoutes(mux *http.ServeMux, cfg *config.Config, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	mux.HandleFunc("/api/topology/test-query", handlers.TopologyTestQueryHandler(cfg.AbsConfigPath, cfg.RouterAPIURL, store))
	log.Printf("Topology Test Query API endpoint registered: /api/topology/test-query (Router API: %s)", cfg.RouterAPIURL)
}

func registerMLPipelineRoutes(mux *http.ServeMux, cfg *config.Config, wf *workflowstore.Store) {
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

	mux.HandleFunc("/api/ml-pipeline/jobs", mlHandler.ListJobsHandler())
	mux.HandleFunc("/api/ml-pipeline/jobs/", mlHandler.GetJobHandler())
	mux.HandleFunc("/api/ml-pipeline/benchmark", mlHandler.RunBenchmarkHandler())
	mux.HandleFunc("/api/ml-pipeline/train", mlHandler.RunTrainHandler())
	mux.HandleFunc("/api/ml-pipeline/config", mlHandler.GenerateConfigHandler())
	mux.HandleFunc("/api/ml-pipeline/download/", mlHandler.DownloadOutputHandler())
	mux.HandleFunc("/api/ml-pipeline/stream/", mlHandler.StreamProgressHandler())
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
