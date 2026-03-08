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
)

func registerEvaluationRoutes(mux *http.ServeMux, cfg *config.Config) {
	if !cfg.EvaluationEnabled {
		log.Printf("Evaluation feature disabled")
		return
	}

	mux.HandleFunc("/api/evaluation/datasets", handlers.GetDatasetsHandler())
	log.Printf("Evaluation datasets endpoint registered: /api/evaluation/datasets")

	evalHandler, err := newEvaluationHandler(cfg)
	if err != nil {
		log.Printf("Warning: failed to initialize evaluation database: %v (other evaluation endpoints disabled)", err)
		return
	}

	registerEvaluationTaskRoutes(mux, evalHandler)
	registerEvaluationExecutionRoutes(mux, evalHandler)
	log.Printf("Evaluation API endpoints registered: /api/evaluation/*")
}

func newEvaluationHandler(cfg *config.Config) (*handlers.EvaluationHandler, error) {
	projectRoot := resolveEvaluationProjectRoot(cfg)
	log.Printf("Evaluation project root: %s", projectRoot)

	evalDB, err := evaluation.NewDB(cfg.EvaluationDBPath)
	if err != nil {
		return nil, err
	}

	runner := evaluation.NewRunner(evaluation.RunnerConfig{
		DB:            evalDB,
		ProjectRoot:   projectRoot,
		PythonPath:    cfg.PythonPath,
		ResultsDir:    cfg.EvaluationResultsDir,
		MaxConcurrent: 10,
	})
	return handlers.NewEvaluationHandler(evalDB, runner, cfg.ReadonlyMode, cfg.RouterAPIURL, cfg.EnvoyURL), nil
}

func resolveEvaluationProjectRoot(cfg *config.Config) string {
	if _, err := os.Stat(filepath.Join(cfg.ConfigDir, "bench")); err == nil {
		return cfg.ConfigDir
	}
	return filepath.Dir(cfg.ConfigDir)
}

func registerEvaluationTaskRoutes(mux *http.ServeMux, evalHandler *handlers.EvaluationHandler) {
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
}

func registerEvaluationExecutionRoutes(mux *http.ServeMux, evalHandler *handlers.EvaluationHandler) {
	mux.HandleFunc("/api/evaluation/run", evalHandler.RunTaskHandler())
	mux.HandleFunc("/api/evaluation/cancel/", evalHandler.CancelTaskHandler())
	mux.HandleFunc("/api/evaluation/stream/", evalHandler.StreamProgressHandler())
	mux.HandleFunc("/api/evaluation/results/", evalHandler.GetResultsHandler())
	mux.HandleFunc("/api/evaluation/export/", evalHandler.ExportResultsHandler())
	mux.HandleFunc("/api/evaluation/history", evalHandler.GetHistoryHandler())
}
