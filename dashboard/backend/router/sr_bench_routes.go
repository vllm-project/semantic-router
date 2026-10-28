package router

import (
	"log"
	"net/http"
	"os"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

func registerSRBenchRoutes(mux routeRegistrar, cfg *config.Config) {
	// Retired paths must not fall through to the generic embedded API proxy.
	// There are no legacy jobs, aliases, or migration handlers.
	notFound := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Cache-Control", "private, no-store")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"message":"Endpoint not found"}}`))
	}
	registerRouteFunc(mux, auth.ProtectedRoute("/api/evaluation", auth.PermEvalRead, auth.SensitivityOperational, auth.ResourceOwnerEvaluation, http.MethodGet), notFound)
	registerRouteFunc(mux, auth.ProtectedRoute("/api/evaluation/", auth.PermEvalRead, auth.SensitivityOperational, auth.ResourceOwnerEvaluation, http.MethodGet), notFound)

	token := os.Getenv(cfg.SRBenchTokenEnv)
	handler, err := handlers.NewSRBenchHandler(cfg.SRBenchURL, token, cfg.ReadonlyMode)
	if err != nil {
		log.Printf("sr-bench proxy configuration is invalid")
		registerRouteGroup(mux, srBenchRouteContracts(), http.HandlerFunc(notFound))
		return
	}
	registerRouteGroup(mux, srBenchRouteContracts(), handler)
	log.Printf("sr-bench 1.0 API registered; workers are owned by the independent service")
}
