package router

import (
	"log"
	"net/http"
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/training"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

func setupTrainingRoutes(mux routeRegistrar, cfg *config.Config, store *workflowstore.Store) {
	// The canonical API shares the existing durable workflow volume. It has no
	// trainer dependency; submitted runs stay pending until an executor is attached.
	service, err := training.New(store, filepath.Join(filepath.Dir(cfg.WorkflowDBPath), "training-files"))
	if err != nil {
		log.Fatalf("training files: %v", err)
	}
	registerTrainingRoutes(mux, service)
}

func registerTrainingRoutes(mux routeRegistrar, s *training.Service) {
	const base = c.APIBasePath
	h := handlers.NewTrainingHandler(s)
	read := func(path string, handler http.HandlerFunc) {
		registerRouteFunc(mux, auth.ProtectedRoute(base+path, auth.PermMlPipeline, auth.SensitivitySecret, auth.ResourceOwnerML, http.MethodGet), handler)
	}
	post := func(path string, handler http.HandlerFunc) {
		registerRouteFunc(mux, auth.ProtectedStreamingMutationRoute(base+path, auth.PermMlPipeline, "training."+path, auth.SensitivitySensitive, auth.ResourceOwnerML, handlers.TrainingJSONMaxBytes, http.MethodPost), handler)
	}
	registerRouteFunc(mux, auth.ProtectedStreamingMutationRoute(base+"/uploads", auth.PermMlPipeline, "training.upload", auth.SensitivitySensitive, auth.ResourceOwnerML, training.MaxFileBytes, http.MethodPost), h.Upload)
	post("/data-assets", handlers.TrainingCreate(201, s.CreateAsset))
	post("/data-snapshots", handlers.TrainingCreate(201, s.CreateSnapshot))
	post("/experiments", handlers.TrainingCreate(201, s.CreateExperiment))
	runs := http.NewServeMux()
	runs.HandleFunc("GET "+base+"/runs", h.ListRuns)
	runs.HandleFunc("POST "+base+"/runs", handlers.TrainingCreate(202, s.Submit))
	runContract := auth.ProtectedStreamingMutationRoute(base+"/runs", auth.PermMlPipeline, "training.submit", auth.SensitivitySensitive, auth.ResourceOwnerML, handlers.TrainingJSONMaxBytes, http.MethodPost)
	runContract.Policies = append(runContract.Policies, auth.ReadPolicy(http.MethodGet, auth.PermMlPipeline, auth.SensitivitySecret, auth.ResourceOwnerML))
	registerRoute(mux, runContract, runs)
	post("/runs/validate", handlers.TrainingCreate(200, s.Validate))
	post("/runs/compare", handlers.TrainingCreate(200, s.Compare))
	post("/runs/{id}/cancel", h.Action(s.Cancel))
	post("/runs/{id}/retry", h.Action(s.Retry))
	for _, kind := range []string{"data-assets", "data-snapshots", "experiments", "runs", "artifacts", "artifact-variants", "evaluations"} {
		read("/"+kind+"/{id}", h.Get(kind))
	}
	read("/runs/{id}/events", h.Events)
	read("/artifacts/{id}/variants", h.Variants)
	read("/files/{id}", h.Download)
}
