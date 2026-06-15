package router

import (
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

func registerWizMapRoutes(mux *http.ServeMux, cfg *config.Config) {
	handler := handlers.WizMapStaticHandler(cfg.StaticDir)
	mux.HandleFunc("/embedded/wizmap", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		handler(w, r)
	})
	mux.HandleFunc("/embedded/wizmap/", func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		handler(w, r)
	})
	log.Printf("WizMap static app registered at /embedded/wizmap/")
}
