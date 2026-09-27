package router

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

func registerWizMapRoutes(mux routeRegistrar, cfg *config.Config) {
	handler := handlers.WizMapStaticHandler(cfg.StaticDir)
	registerRouteFunc(mux, auth.ProtectedRoute("/embedded/wizmap", auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		handler(w, r)
	})
	registerRouteFunc(mux, auth.ProtectedRoute("/embedded/wizmap/", auth.PermConfigRead, auth.SensitivitySensitive, auth.ResourceOwnerConfig, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		handler(w, r)
	})
	registerRouteFunc(mux, auth.PublicRoute("/embedded/wizmap/assets/", http.MethodGet), handler)
	log.Printf("WizMap static app registered at /embedded/wizmap/")
}
