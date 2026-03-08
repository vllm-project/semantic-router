package router

import (
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/mlpipeline"
)

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
