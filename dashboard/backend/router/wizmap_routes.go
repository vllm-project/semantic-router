package router

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

func registerWizMapRoutes(routes *auth.PolicyMux, cfg *config.Config) {
	handler := handlers.WizMapStaticHandler(cfg.StaticDir)
	app := func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		handler(w, r)
	}
	routes.HandleGroup([]auth.RouteContract{
		auth.ProtectedRoute("/embedded/wizmap", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet, http.MethodHead),
		auth.ProtectedRoute("/embedded/wizmap/", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet, http.MethodHead),
	}, http.HandlerFunc(app))
	// Static assets carry no data; the app shell above is what needs a session.
	routes.HandleFunc(auth.PublicRoute("/embedded/wizmap/assets/", http.MethodGet, http.MethodHead), handler)
	log.Printf("WizMap static app registered at /embedded/wizmap/")
}
