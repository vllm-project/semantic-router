package router

import (
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func registerCoreRoutes(mux *http.ServeMux, cfg *config.Config) {
	registerHealthAndSetupRoutes(mux, cfg)
	registerConfigRoutes(mux, cfg)
	registerToolRoutes(mux, cfg)
	registerStatusRoutes(mux, cfg)
	registerTopologyRoutes(mux, cfg)
}

func registerHealthAndSetupRoutes(mux *http.ServeMux, cfg *config.Config) {
	mux.HandleFunc("/healthz", handlers.HealthCheck)
	mux.HandleFunc("/api/settings", handlers.SettingsHandler(cfg))
	mux.HandleFunc("/api/setup/state", handlers.SetupStateHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/setup/import-remote", handlers.SetupImportRemoteHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/setup/validate", handlers.SetupValidateHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/setup/activate", handlers.SetupActivateHandler(cfg.AbsConfigPath, cfg.ReadonlyMode, cfg.ConfigDir))
}

func registerConfigRoutes(mux *http.ServeMux, cfg *config.Config) {
	mux.HandleFunc("/api/router/config/all", handlers.ConfigHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/yaml", handlers.ConfigYAMLHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/update", handlers.UpdateConfigHandler(cfg.AbsConfigPath, cfg.ReadonlyMode, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/deploy/preview", handlers.DeployPreviewHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/deploy", handlers.DeployHandler(cfg.AbsConfigPath, cfg.ReadonlyMode, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/rollback", handlers.RollbackHandler(cfg.AbsConfigPath, cfg.ReadonlyMode, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/versions", handlers.ConfigVersionsHandler(cfg.AbsConfigPath))
	log.Printf("Config API endpoints registered: /api/router/config/all, /api/router/config/yaml, /api/router/config/update, /api/router/config/deploy, /api/router/config/deploy/preview, /api/router/config/rollback, /api/router/config/versions")

	mux.HandleFunc("/api/router/config/global", handlers.RouterDefaultsHandler(cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/global/update", handlers.UpdateRouterDefaultsHandler(cfg.ConfigDir, cfg.ReadonlyMode))
	mux.HandleFunc("/api/router/config/global/raw", handlers.GlobalConfigYAMLHandler(cfg.AbsConfigPath))
	mux.HandleFunc("/api/router/config/global/raw/update", handlers.UpdateGlobalConfigYAMLHandler(cfg.AbsConfigPath, cfg.ReadonlyMode, cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/defaults", handlers.RouterDefaultsHandler(cfg.ConfigDir))
	mux.HandleFunc("/api/router/config/defaults/update", handlers.UpdateRouterDefaultsHandler(cfg.ConfigDir, cfg.ReadonlyMode))
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

	mux.HandleFunc("/api/tools/fetch-raw", handlers.FetchRawHandler())
	log.Printf("Fetch Raw API endpoint registered: /api/tools/fetch-raw")
}

func resolveToolsDBPath(cfg *config.Config) string {
	toolsDBPath := filepath.Join(cfg.ConfigDir, "config", "tools_db.json")
	parsedCfg, err := routerconfig.Parse(cfg.AbsConfigPath)
	if err != nil {
		log.Printf("Warning: failed to parse config for tools_db_path, use the default path %s: %v", toolsDBPath, err)
		return toolsDBPath
	}
	if parsedCfg.ToolSelection.Tools.ToolsDBPath != "" {
		return parsedCfg.ToolSelection.Tools.ToolsDBPath
	}
	return toolsDBPath
}

func registerStatusRoutes(mux *http.ServeMux, cfg *config.Config) {
	mux.HandleFunc("/api/status", handlers.StatusHandler(cfg.RouterAPIURL, cfg.ConfigDir))
	log.Printf("Status API endpoint registered: /api/status")

	mux.HandleFunc("/api/logs", handlers.LogsHandler(cfg.RouterAPIURL))
	log.Printf("Logs API endpoint registered: /api/logs")
}

func registerTopologyRoutes(mux *http.ServeMux, cfg *config.Config) {
	mux.HandleFunc("/api/topology/test-query", handlers.TopologyTestQueryHandler(cfg.AbsConfigPath, cfg.RouterAPIURL))
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
	projectRoot := filepath.Dir(cfg.ConfigDir)
	if _, err := os.Stat(filepath.Join(cfg.ConfigDir, "bench")); err == nil {
		return cfg.ConfigDir
	}
	return projectRoot
}

func registerMLPipelineRoutes(mux *http.ServeMux, cfg *config.Config) {
	if !cfg.MLPipelineEnabled {
		log.Printf("ML Pipeline feature disabled")
		return
	}

	trainingDir := resolveMLTrainingDir(cfg)
	mlRunner := mlpipeline.NewRunner(mlpipeline.RunnerConfig{
		DataDir:      cfg.MLPipelineDataDir,
		TrainingDir:  trainingDir,
		PythonPath:   cfg.PythonPath,
		MLServiceURL: cfg.MLServiceURL,
	})
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
