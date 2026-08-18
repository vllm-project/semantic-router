package router

import (
	"context"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
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

func registerCoreRoutes(mux *http.ServeMux, cfg *config.Config, routeOptions ...coreRouteOptions) {
	options := coreRouteOptions{}
	if len(routeOptions) > 0 {
		options = routeOptions[0]
	}
	store := selectedRecipeStore(cfg, []*recipe.Store{options.recipeStore})
	registerHealthAndSetupRoutes(mux, cfg)
	registerConfigRoutes(mux, cfg, configRouteOptions{
		credentialStore:          store,
		modelVerificationAuditor: options.modelVerificationAuditor,
	})
	registerToolRoutes(mux, cfg)
	registerStatusRoutes(mux, cfg, store)
	registerTopologyRoutes(mux, cfg, store)
	registerRecipeRoutes(mux, cfg, store)
	registerSecurityPolicyRoutes(mux, cfg)
}

func registerRecipeRoutes(mux *http.ServeMux, cfg *config.Config, stores ...*recipe.Store) {
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

func registerSecurityPolicyRoutes(mux *http.ServeMux, cfg *config.Config) {
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	handlers.SetSecurityPolicyConfigPaths(cfg.AbsConfigPath, cfg.ConfigDir)
	mux.HandleFunc("/api/security/policy", func(w http.ResponseWriter, r *http.Request) {
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
	mux.HandleFunc("/api/security/policy/preview", func(w http.ResponseWriter, r *http.Request) {
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

func registerHealthAndSetupRoutes(mux *http.ServeMux, cfg *config.Config) {
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	mux.HandleFunc("/healthz", handlers.HealthCheck)
	mux.HandleFunc("/api/settings", handlers.SettingsHandler(cfg))
	mux.HandleFunc("/api/setup/state", handlers.SetupStateHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/setup/import-remote", handlers.SetupImportRemoteHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/setup/validate", handlers.SetupValidateHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/setup/activate", handlers.SetupActivateHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/setup/presets", handlers.PresetsHandler())
	mux.HandleFunc("/api/setup/presets/delta", handlers.PresetDeltaHandler())
}

func registerConfigRoutes(mux *http.ServeMux, cfg *config.Config, routeOptions ...configRouteOptions) {
	options := configRouteOptions{}
	if len(routeOptions) > 0 {
		options = routeOptions[0]
	}
	runtimeConfigReadonly := cfg.ReadonlyMode || !cfg.RuntimeConfigWritable
	mux.HandleFunc("/api/models/catalog", handlers.ModelCatalogHandler(handlers.NewCLIModelCatalogSource(cfg.PythonPath)))
	mux.HandleFunc("/api/models/verify", handlers.ModelVerificationHandler(cfg.AbsConfigPath, options.modelVerificationAuditor))
	mux.HandleFunc("/api/router/config/all", handlers.ConfigHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/yaml", handlers.ConfigYAMLHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/update", handlers.UpdateConfigHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/deploy/preview", handlers.DeployPreviewHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/deploy", handlers.DeployHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/rollback", handlers.RollbackHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/versions", handlers.ConfigVersionsHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/deployments", handlers.ConfigDeploymentsHandler())
	mux.HandleFunc("/api/router/config/deployments/", handlers.ConfigDeploymentDetailHandler())
	mux.HandleFunc("/api/router/config/active-projection", handlers.ActiveConfigProjectionHandler())
	mux.HandleFunc("/api/router/config/nl/verify", handlers.BuilderNLVerifyHandler(cfg.AbsConfigPath, cfg.EnvoyURL))
	mux.HandleFunc("/api/router/config/nl/generate/stream", handlers.BuilderNLGenerateStreamHandler(cfg.AbsConfigPath, cfg.EnvoyURL))
	mux.HandleFunc("/api/router/config/nl/generate", handlers.BuilderNLGenerateHandler(cfg.AbsConfigPath, cfg.EnvoyURL))
	log.Printf("Config API endpoints registered: /api/models/catalog, /api/models/verify, /api/router/config/all, /api/router/config/yaml, /api/router/config/update, /api/router/config/nl/verify, /api/router/config/nl/generate/stream, /api/router/config/nl/generate, /api/router/config/deploy, /api/router/config/deploy/preview, /api/router/config/rollback, /api/router/config/versions, /api/router/config/deployments, /api/router/config/active-projection")

	mux.HandleFunc("/api/router/config/global", handlers.RouterDefaultsHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/global/update", handlers.UpdateRouterDefaultsHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/global/raw", handlers.GlobalConfigYAMLHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/global/raw/update", handlers.UpdateGlobalConfigYAMLHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/defaults", handlers.RouterDefaultsHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/defaults/update", handlers.UpdateRouterDefaultsHandler(cfg.AbsConfigPath, runtimeConfigReadonly, cfg.ConfigDir))
	store := selectedRecipeStore(cfg, []*recipe.Store{options.credentialStore})
	mux.HandleFunc("/api/router/config/kbs", handlers.RouterClassifierProxyHandler(cfg.RouterAPIURL, cfg.ReadonlyMode, store))
	mux.HandleFunc("/api/router/config/kbs/", handlers.RouterClassifierProxyHandler(cfg.RouterAPIURL, cfg.ReadonlyMode, store))
	log.Printf("Global config API endpoints registered: /api/router/config/global, /api/router/config/global/update, /api/router/config/global/raw, /api/router/config/global/raw/update (legacy aliases: /api/router/config/defaults, /api/router/config/defaults/update)")
}

func registerToolRoutes(mux *http.ServeMux, cfg *config.Config) {
	toolsDBPath := resolveToolsDBPath(cfg)
	mux.HandleFunc("/api/tools-db", handlers.ToolsDBHandler(toolsDBPath))
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

func registerStatusRoutes(mux *http.ServeMux, cfg *config.Config, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	mux.HandleFunc("/api/status", handlers.StatusHandler(cfg.RouterAPIURL, cfg.ConfigDir, store))
	log.Printf("Status API endpoint registered: /api/status")

	mux.HandleFunc("/api/logs", handlers.LogsHandler(cfg.RouterAPIURL))
	log.Printf("Logs API endpoint registered: /api/logs")
}

func registerTopologyRoutes(mux *http.ServeMux, cfg *config.Config, credentialProvider ...*recipe.Store) {
	store := selectedRecipeStore(cfg, credentialProvider)
	mux.HandleFunc("/api/topology/test-query", handlers.TopologyTestQueryHandler(cfg.AbsConfigPath, cfg.RouterAPIURL, store))
	log.Printf("Topology Test Query API endpoint registered: /api/topology/test-query (Router API: %s)", cfg.RouterAPIURL)
}

func registerEvaluationRoutes(mux *http.ServeMux, cfg *config.Config) {
	if !cfg.EvaluationEnabled {
		log.Printf("Evaluation feature disabled")
		return
	}

	mux.HandleFunc("/api/evaluation/datasets", handlers.GetDatasetsHandler())
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

	mux.HandleFunc("/api/evaluation/tasks", func(w http.ResponseWriter, r *http.Request) {
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
	mux.HandleFunc("/api/evaluation/tasks/", func(w http.ResponseWriter, r *http.Request) {
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
	mux.HandleFunc("/api/evaluation/run", evalHandler.RunTaskHandler())
	mux.HandleFunc("/api/evaluation/cancel/", evalHandler.CancelTaskHandler())
	mux.HandleFunc("/api/evaluation/stream/", evalHandler.StreamProgressHandler())
	mux.HandleFunc("/api/evaluation/results/", evalHandler.GetResultsHandler())
	mux.HandleFunc("/api/evaluation/export/", evalHandler.ExportResultsHandler())
	mux.HandleFunc("/api/evaluation/history", evalHandler.GetHistoryHandler())
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
