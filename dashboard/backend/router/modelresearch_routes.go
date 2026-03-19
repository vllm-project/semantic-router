package router

import (
	"log"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/modelresearch"
)

func registerModelResearchRoutes(mux *http.ServeMux, cfg *config.Config) {
	repoRoot := filepath.Dir(cfg.ConfigDir)
	manager, err := modelresearch.NewManager(modelresearch.ManagerConfig{
		BaseDir:             filepath.Join(cfg.ConfigDir, ".vllm-sr", "model-research"),
		RepoRoot:            repoRoot,
		PythonPath:          cfg.PythonPath,
		DefaultAPIBase:      cfg.RouterAPIURL,
		DefaultRequestModel: "MoM",
		Platform:            cfg.Platform,
	})
	if err != nil {
		log.Printf("Warning: failed to initialize model research manager: %v", err)
		return
	}

	handler := handlers.NewModelResearchHandler(manager)
	mux.HandleFunc("/api/model-research/recipes", handler.RecipesHandler())
	mux.HandleFunc("/api/model-research/campaigns", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			handler.ListCampaignsHandler().ServeHTTP(w, r)
		case http.MethodPost:
			handler.CreateCampaignHandler().ServeHTTP(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	})
	mux.HandleFunc("/api/model-research/campaigns/", func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/events"):
			handler.StreamEventsHandler().ServeHTTP(w, r)
		case strings.HasSuffix(r.URL.Path, "/stop"):
			handler.StopCampaignHandler().ServeHTTP(w, r)
		default:
			handler.GetCampaignHandler().ServeHTTP(w, r)
		}
	})
	log.Printf("Model Research API endpoints registered: /api/model-research/*")
}
