package router

import (
	"log"
	"net/http"

	auth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

func registerWizMapRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	handler := handlers.WizMapStaticHandler(cfg.StaticDir)
	routes.HandleFunc(auth.ProtectedRoute("/embedded/wizmap", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		handler(w, r)
	})
	routes.HandleFunc(auth.ProtectedRoute("/embedded/wizmap/", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet), func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		handler(w, r)
	})
	routes.HandleFunc(auth.PublicRoute("/embedded/wizmap/assets/", http.MethodGet), handler)
	log.Printf("WizMap static app registered at /embedded/wizmap/")
}
