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
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

type evaluationAccessProvider interface {
	handlers.EvaluationScopeResolver
	handlers.EvaluationRunAuthorizer
}

func registerCoreRoutes(mux *http.ServeMux, cfg *config.Config, statusHandler http.HandlerFunc) {
	registerHealthRoutes(mux, cfg)
	registerStatusRoutes(mux, cfg, statusHandler)
}

func registerHealthRoutes(mux *http.ServeMux, cfg *config.Config) {
	mux.HandleFunc("/healthz", handlers.HealthCheck)
	mux.HandleFunc("/api/settings", handlers.SettingsHandler(cfg))
}

func registerStatusRoutes(
	mux *http.ServeMux,
	cfg *config.Config,
	statusHandler http.HandlerFunc,
) {
	if statusHandler == nil {
		statusHandler = handlers.StatusHandler(cfg.RouterAPIURL, nil)
	}
	mux.HandleFunc("/api/status", statusHandler)
	log.Printf("Status API endpoint registered: /api/status")

	mux.HandleFunc("/api/logs", handlers.LogsHandler(cfg.RouterAPIURL))
	log.Printf("Logs API endpoint registered: /api/logs")
}

func registerEvaluationRoutes(
	mux *http.ServeMux,
	cfg *config.Config,
	accessProvider evaluationAccessProvider,
) {
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
	evalHandler := handlers.NewEvaluationHandler(
		evalDB,
		runner,
		cfg.ReadonlyMode,
		cfg.RouterAPIURL,
		cfg.EnvoyURL,
		accessProvider,
		accessProvider,
	)

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
