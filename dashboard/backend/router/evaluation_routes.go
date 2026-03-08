package router

import (
	"log"
	"net/http"
	"os"
	"path/filepath"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/evaluation"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

func registerEvaluationRoutes(mux *http.ServeMux, app *backendapp.App) {
	cfg := app.Config
	access := newRouteAccess(app)
	if !cfg.EvaluationEnabled {
		log.Printf("Evaluation feature disabled")
		return
	}

	mux.Handle("/api/evaluation/datasets", access.viewer(handlers.GetDatasetsHandler()))
	log.Printf("Evaluation datasets endpoint registered: /api/evaluation/datasets")

	evalHandler, err := newEvaluationHandler(cfg)
	if err != nil {
		log.Printf("Warning: failed to initialize evaluation database: %v (other evaluation endpoints disabled)", err)
		return
	}

	registerEvaluationTaskRoutes(mux, access, evalHandler)
	registerEvaluationExecutionRoutes(mux, access, evalHandler)
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

func registerEvaluationTaskRoutes(mux *http.ServeMux, access routeAccess, evalHandler *handlers.EvaluationHandler) {
	listTasks := access.viewer(evalHandler.ListTasksHandler())
	createTask := access.operator(evalHandler.CreateTaskHandler())
	getTask := access.viewer(evalHandler.GetTaskHandler())
	deleteTask := access.operator(evalHandler.DeleteTaskHandler())

	mux.HandleFunc("/api/evaluation/tasks", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		switch r.Method {
		case http.MethodGet:
			listTasks.ServeHTTP(w, r)
		case http.MethodPost:
			createTask.ServeHTTP(w, r)
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
			getTask.ServeHTTP(w, r)
		case http.MethodDelete:
			deleteTask.ServeHTTP(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
}

func registerEvaluationExecutionRoutes(mux *http.ServeMux, access routeAccess, evalHandler *handlers.EvaluationHandler) {
	mux.Handle("/api/evaluation/run", access.operator(evalHandler.RunTaskHandler()))
	mux.Handle("/api/evaluation/cancel/", access.operator(evalHandler.CancelTaskHandler()))
	mux.Handle("/api/evaluation/stream/", access.viewer(evalHandler.StreamProgressHandler()))
	mux.Handle("/api/evaluation/results/", access.viewer(evalHandler.GetResultsHandler()))
	mux.Handle("/api/evaluation/export/", access.operator(evalHandler.ExportResultsHandler()))
	mux.Handle("/api/evaluation/history", access.viewer(evalHandler.GetHistoryHandler()))
}
