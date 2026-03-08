package router

import (
	"log"
	"net/http"
	"os"
	"path/filepath"

	backendapp "github.com/vllm-project/semantic-router/dashboard/backend/app"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
)

func registerMLPipelineRoutes(mux *http.ServeMux, app *backendapp.App) {
	cfg := app.Config
	access := newRouteAccess(app)
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

	mux.Handle("/api/ml-pipeline/jobs", access.viewer(mlHandler.ListJobsHandler()))
	mux.Handle("/api/ml-pipeline/jobs/", access.viewer(mlHandler.GetJobHandler()))
	mux.Handle("/api/ml-pipeline/benchmark", access.operator(mlHandler.RunBenchmarkHandler()))
	mux.Handle("/api/ml-pipeline/train", access.operator(mlHandler.RunTrainHandler()))
	mux.Handle("/api/ml-pipeline/config", access.operator(mlHandler.GenerateConfigHandler()))
	mux.Handle("/api/ml-pipeline/download/", access.viewer(mlHandler.DownloadOutputHandler()))
	mux.Handle("/api/ml-pipeline/stream/", access.viewer(mlHandler.StreamProgressHandler()))
	log.Printf("ML Pipeline API endpoints registered: /api/ml-pipeline/*")
	logMLTrainingDir(trainingDir)
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

func logMLTrainingDir(trainingDir string) {
	if trainingDir != "" {
		log.Printf("ML Training scripts directory: %s", trainingDir)
		return
	}
	log.Printf("Warning: ML training scripts directory not configured (set ML_TRAINING_DIR)")
}
