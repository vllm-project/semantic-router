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

const (
	maxConfigBodyBytes  = 16 << 20
	maxProbeBodyBytes   = 2 << 20
	maxToolBodyBytes    = 256 << 10
	maxMLBodyBytes      = 2 << 20
	maxCompactBodyBytes = 64 << 10
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
	registerStatusRoutes(routes, cfg, options.statusHandler, store)
	registerTopologyRoutes(routes, cfg, store)
	registerRecipeRoutes(routes, cfg, store)
}

func configRead(pattern string, sensitivity auth.Sensitivity) auth.RouteContract {
	return auth.ProtectedRoute(pattern, auth.PermConfigRead, sensitivity, auth.ResourceOwnerConfig, http.MethodGet)
}

func registerRecipeRoutes(routes *auth.PolicyMux, cfg *config.Config, stores ...*recipe.Store) {
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
	routes.HandleFunc(configRead("/api/recipe", auth.SensitivityOperational), handler.Descriptor)
	routes.HandleFunc(configRead("/api/recipe/probes", auth.SensitivityOperational), handler.Probes)
	// The probe handler trims a trailing slash itself, so each probe path is
	// registered with its slash alias.
	probeContracts := make([]auth.RouteContract, 0, 6)
	for _, alias := range []string{"", "/{$}"} {
		probeContracts = append(probeContracts,
			configRead("/api/recipe/probes/{decision}/{variant}"+alias, auth.SensitivityOperational),
			auth.ProtectedBoundedRoute("/api/recipe/probes/{decision}/{variant}/run-plan"+alias, auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig, maxProbeBodyBytes, http.MethodPost),
			auth.ProtectedBoundedRoute("/api/recipe/probes/{decision}/{variant}/validate"+alias, auth.PermTopologyRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig, maxProbeBodyBytes, http.MethodPost),
		)
	}
	routes.HandleGroup(probeContracts, http.HandlerFunc(handler.ProbeAction))
	routes.HandleGroup([]auth.RouteContract{
		configRead("/api/recipe/packages", auth.SensitivityOperational),
		configRead("/api/recipe/packages/", auth.SensitivityOperational),
	}, http.HandlerFunc(handler.Packages))
	routes.HandleGroup(recipePackageMutationContracts("/api/recipe/import", auth.PermConfigWrite, "recipe.import"), http.HandlerFunc(handler.ImportPackage))
	routes.HandleGroup(recipePackageMutationContracts("/api/recipe/activate", auth.PermConfigDeploy, "recipe.activate"), http.HandlerFunc(handler.ActivatePackage))
	routes.HandleGroup(recipePackagePreviewContracts("/api/recipe/activate/preview"), http.HandlerFunc(handler.PreviewPackageActivation))
	routes.HandleGroup(recipePackageMutationContracts("/api/recipe/deactivate", auth.PermConfigDeploy, "recipe.deactivate"), http.HandlerFunc(handler.DeactivatePackage))
	routes.HandleGroup(recipePackagePreviewContracts("/api/recipe/deactivate/preview"), http.HandlerFunc(handler.PreviewPackageDeactivation))
	log.Printf("Active Recipe API endpoints registered: /api/recipe, /api/recipe/probes/*, /api/recipe/packages, /api/recipe/import, /api/recipe/activate/preview, /api/recipe/activate, /api/recipe/deactivate/preview, /api/recipe/deactivate")
}

// Package handlers own the whole subtree under the mutation permission and
// answer a no-store 404 for anything but the canonical path or its
// trailing-slash alias.
func recipePackageMutationContracts(pattern, permission, action string) []auth.RouteContract {
	return []auth.RouteContract{
		auth.ProtectedMutationRoute(pattern, permission, action, auth.SensitivitySecret, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost),
		auth.ProtectedMutationRoute(pattern+"/", permission, action, auth.SensitivitySecret, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost),
	}
}

func recipePackagePreviewContracts(pattern string) []auth.RouteContract {
	return []auth.RouteContract{
		auth.ProtectedBoundedRoute(pattern, auth.PermConfigDeploy, auth.SensitivitySensitive, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost),
		auth.ProtectedBoundedRoute(pattern+"/", auth.PermConfigDeploy, auth.SensitivitySensitive, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost),
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

func registerHealthAndSetupRoutes(routes *auth.PolicyMux, cfg *config.Config, setupResolver *setupmode.Resolver) {
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	routes.HandleFunc(auth.PublicRoute("/healthz", http.MethodGet, http.MethodHead), handlers.HealthCheck)
	routes.HandleFunc(configRead("/api/settings", auth.SensitivityOperational), handlers.SettingsHandler(cfg, setupResolver))
	routes.HandleFunc(auth.PublicRoute("/api/setup/state", http.MethodGet), handlers.SetupStateHandler(cfg.AbsConfigPath, setupResolver))
	routes.HandleFunc(
		auth.ProtectedBoundedRoute("/api/setup/import-remote", auth.PermConfigWrite, auth.SensitivitySensitive, auth.ResourceOwnerConfig, maxCompactBodyBytes, http.MethodPost),
		handlers.SetupImportRemoteHandler(cfg.AbsConfigPath, setupResolver),
	)
	routes.HandleFunc(
		auth.ProtectedBoundedRoute("/api/setup/validate", auth.PermConfigWrite, auth.SensitivitySensitive, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost),
		handlers.SetupValidateHandler(cfg.AbsConfigPath, setupResolver),
	)
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/setup/activate", auth.PermConfigWrite, "setup.activate", auth.SensitivitySecret, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost),
		handlers.SetupActivateHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir, setupResolver),
	)
	routes.HandleFunc(configRead("/api/setup/presets", auth.SensitivityOperational), handlers.PresetsHandler())
	routes.HandleFunc(
		auth.ProtectedBoundedRoute("/api/setup/presets/delta", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, maxProbeBodyBytes, http.MethodPost),
		handlers.PresetDeltaHandler(),
	)
}

func registerConfigRoutes(routes *auth.PolicyMux, cfg *config.Config, routeOptions ...configRouteOptions) {
	if err := handlers.RestrictExistingConfigSnapshots(cfg.ConfigDir); err != nil {
		log.Printf("Warning: could not restrict existing config snapshots: %v", err)
	}
	options := configRouteOptions{}
	if len(routeOptions) > 0 {
		options = routeOptions[0]
	}
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	store := selectedRecipeStore(cfg, []*recipe.Store{options.credentialStore})
	configWrite := func(pattern, action string, methods ...string) auth.RouteContract {
		return auth.ProtectedMutationRoute(pattern, auth.PermConfigWrite, action, auth.SensitivitySecret, auth.ResourceOwnerConfig, maxConfigBodyBytes, methods...)
	}
	routes.HandleFunc(configRead("/api/models/catalog", auth.SensitivityOperational), handlers.ModelCatalogHandler(handlers.NewPackagedModelCatalogSource(cfg.PythonPath)))
	routes.HandleFunc(
		auth.ProtectedBoundedRoute("/api/models/discover", auth.PermConfigWrite, auth.SensitivitySensitive, auth.ResourceOwnerConfig, maxProbeBodyBytes, http.MethodPost),
		handlers.ModelDiscoveryHandler(nil),
	)
	routes.HandleFunc(
		auth.ProtectedDelegatedMutationRoute("/api/models/verify", auth.PermEvalRun, "model.inference_verify", auth.SensitivitySensitive, auth.ResourceOwnerInference, maxProbeBodyBytes, http.MethodPost),
		handlers.ModelVerificationHandler(cfg.AbsConfigPath, options.modelVerificationAuditor),
	)
	routes.HandleFunc(configRead("/api/router/config/all", auth.SensitivitySecret), handlers.ConfigHandler(cfg.AbsConfigPath))
	routes.HandleFunc(configRead("/api/router/config/schema", auth.SensitivityOperational), handlers.ConfigSchemaHandler(cfg.RouterAPIURL, store))
	routes.HandleFunc(configRead("/api/router/config/yaml", auth.SensitivitySecret), handlers.ConfigYAMLHandler(cfg.AbsConfigPath))
	routes.HandleFunc(configWrite("/api/router/config/update", "config.update", http.MethodPost, http.MethodPut), handlers.UpdateConfigHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	routes.HandleFunc(
		auth.ProtectedBoundedRoute("/api/router/config/deploy/preview", auth.PermConfigDeploy, auth.SensitivitySensitive, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost),
		handlers.DeployPreviewHandler(cfg.AbsConfigPath),
	)
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/router/config/deploy", auth.PermConfigDeploy, "config.deploy", auth.SensitivitySecret, auth.ResourceOwnerConfig, maxConfigBodyBytes, http.MethodPost),
		handlers.DeployHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir),
	)
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/router/config/rollback", auth.PermConfigDeploy, "config.rollback", auth.SensitivitySecret, auth.ResourceOwnerConfig, maxCompactBodyBytes, http.MethodPost),
		handlers.RollbackHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir),
	)
	routes.HandleFunc(configRead("/api/router/config/versions", auth.SensitivityOperational), handlers.ConfigVersionsHandler(cfg.AbsConfigPath))
	routes.HandleFunc(configRead("/api/router/config/deployments", auth.SensitivityOperational), handlers.ConfigDeploymentsHandler())
	routes.HandleFunc(configRead("/api/router/config/deployments/{version}", auth.SensitivitySensitive), handlers.ConfigDeploymentDetailHandler())
	routes.HandleFunc(configRead("/api/router/config/active-projection", auth.SensitivitySensitive), handlers.ActiveConfigProjectionHandler())
	log.Printf("Config API endpoints registered: /api/models/catalog, /api/models/discover, /api/models/verify, /api/router/config/all, /api/router/config/schema, /api/router/config/yaml, /api/router/config/update, /api/router/config/deploy, /api/router/config/deploy/preview, /api/router/config/rollback, /api/router/config/versions, /api/router/config/deployments, /api/router/config/active-projection")

	routes.HandleFunc(configRead("/api/router/config/global", auth.SensitivitySensitive), handlers.RouterDefaultsHandler(cfg.AbsConfigPath))
	routes.HandleFunc(configWrite("/api/router/config/global/update", "config.global.update", http.MethodPost, http.MethodPut), handlers.UpdateRouterDefaultsHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	routes.HandleFunc(configRead("/api/router/config/global/raw", auth.SensitivitySecret), handlers.GlobalConfigYAMLHandler(cfg.AbsConfigPath))
	routes.HandleFunc(configWrite("/api/router/config/global/raw/update", "config.global_raw.update", http.MethodPost, http.MethodPut), handlers.UpdateGlobalConfigYAMLHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	// Knowledge-base storage is served by the Dashboard classifier proxy, not
	// the generic Router gateway, but shares the gateway contract.
	routes.HandleGroup(
		routerManagementContracts(isKnowledgeBasePath),
		handlers.RouterClassifierProxyHandler(cfg.RouterAPIURL, cfg.ReadonlyMode, store),
	)
	log.Printf("Global config API endpoints registered: /api/router/config/global, /api/router/config/global/update, /api/router/config/global/raw, /api/router/config/global/raw/update")
}

func registerToolRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	routes.HandleFunc(
		auth.ProtectedRoute("/api/tools-db", auth.PermToolsUse, auth.SensitivitySensitive, auth.ResourceOwnerTools, http.MethodGet),
		func(w http.ResponseWriter, r *http.Request) {
			// Configuration saves can change the selected database without restarting
			// Dashboard. Resolve the current canonical path for each refresh.
			handlers.ToolsDBHandler(resolveToolsDBPath(cfg))(w, r)
		},
	)
	log.Printf("Tools DB API endpoint registered: /api/tools-db")

	tool := func(pattern string) auth.RouteContract {
		return auth.ProtectedBoundedRoute(pattern, auth.PermToolsUse, auth.SensitivitySensitive, auth.ResourceOwnerTools, maxToolBodyBytes, http.MethodPost)
	}
	routes.HandleFunc(tool("/api/tools/web-search"), handlers.WebSearchHandler())
	log.Printf("Web Search API endpoint registered: /api/tools/web-search")

	routes.HandleFunc(tool("/api/tools/open-web"), handlers.OpenWebHandler())
	log.Printf("Open Web API endpoint registered: /api/tools/open-web")

	routes.HandleFunc(tool("/api/tools/weather"), handlers.WeatherHandler())
	log.Printf("Weather API endpoint registered: /api/tools/weather")

	routes.HandleFunc(tool("/api/tools/fetch-raw"), handlers.FetchRawHandler())
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

func registerStatusRoutes(routes *auth.PolicyMux, cfg *config.Config, statusHandler http.HandlerFunc, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	if statusHandler == nil {
		statusHandler = handlers.StatusHandler(cfg.RouterAPIURL, cfg.EnvoyURL, cfg.ConfigDir, store)
	}
	// The status summary is the unauthenticated liveness surface the frontend
	// polls before login.
	routes.HandleGroup([]auth.RouteContract{
		auth.PublicRoute("/api/status", http.MethodGet),
		auth.PublicRoute("/api/status/{$}", http.MethodGet),
	}, statusHandler)
	log.Printf("Status API endpoint registered: /api/status")

	routes.HandleFunc(
		auth.ProtectedRoute("/api/logs", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet),
		handlers.LogsHandler(cfg.RouterAPIURL),
	)
	log.Printf("Logs API endpoint registered: /api/logs")
}

func registerTopologyRoutes(routes *auth.PolicyMux, cfg *config.Config, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	routes.HandleFunc(
		auth.ProtectedBoundedRoute("/api/topology/test-query", auth.PermTopologyRead, auth.SensitivitySensitive, auth.ResourceOwnerInference, maxProbeBodyBytes, http.MethodPost),
		handlers.TopologyTestQueryHandler(cfg.AbsConfigPath, cfg.RouterAPIURL, store),
	)
	log.Printf("Topology Test Query API endpoint registered: /api/topology/test-query (Router API: %s)", cfg.RouterAPIURL)
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

	mlRead := func(pattern string) auth.RouteContract {
		return auth.ProtectedRoute(pattern, auth.PermMlPipeline, auth.SensitivitySensitive, auth.ResourceOwnerML, http.MethodGet)
	}
	// Dataset uploads are multipart bodies far larger than a JSON mutation;
	// they stream through the bound and the handlers revalidate before the
	// job starts. The limits match the handlers' ParseMultipartForm sizes.
	mlUpload := func(pattern, action string, maxBodyBytes int64) auth.RouteContract {
		return auth.ProtectedStreamingMutationRoute(pattern, auth.PermMlPipeline, action, auth.SensitivitySensitive, auth.ResourceOwnerML, maxBodyBytes, http.MethodPost)
	}
	routes.HandleFunc(mlRead("/api/ml-pipeline/jobs"), mlHandler.ListJobsHandler())
	routes.HandleGroup([]auth.RouteContract{
		mlRead("/api/ml-pipeline/jobs/{id}"),
		mlRead("/api/ml-pipeline/jobs/{id}/{$}"),
		mlRead("/api/ml-pipeline/jobs/{id}/events"),
	}, mlHandler.GetJobHandler())
	routes.HandleFunc(mlUpload("/api/ml-pipeline/benchmark", "ml.benchmark", handlers.MLBenchmarkUploadMaxBytes), mlHandler.RunBenchmarkHandler())
	routes.HandleFunc(mlUpload("/api/ml-pipeline/train", "ml.train", handlers.MLTrainUploadMaxBytes), mlHandler.RunTrainHandler())
	routes.HandleFunc(
		auth.ProtectedMutationRoute("/api/ml-pipeline/config", auth.PermMlPipeline, "ml.config.generate", auth.SensitivitySensitive, auth.ResourceOwnerML, maxMLBodyBytes, http.MethodPost),
		mlHandler.GenerateConfigHandler(),
	)
	routes.HandleGroup([]auth.RouteContract{
		mlRead("/api/ml-pipeline/download/{id}"),
		mlRead("/api/ml-pipeline/download/{id}/{$}"),
		mlRead("/api/ml-pipeline/download/{id}/{index}"),
	}, mlHandler.DownloadOutputHandler())
	routes.HandleGroup([]auth.RouteContract{
		mlRead("/api/ml-pipeline/stream/{id}"),
		mlRead("/api/ml-pipeline/stream/{id}/{$}"),
	}, mlHandler.StreamProgressHandler())
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
