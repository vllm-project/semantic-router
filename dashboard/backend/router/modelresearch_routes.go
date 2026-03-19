package router

import (
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/modelresearch"
)

func registerModelResearchRoutes(mux *http.ServeMux, cfg *config.Config) {
	repoRoot := resolveModelResearchProjectRoot(cfg)
	manager, err := modelresearch.NewManager(modelresearch.ManagerConfig{
		BaseDir:             filepath.Join(cfg.ConfigDir, ".vllm-sr", "model-research"),
		RepoRoot:            repoRoot,
		PythonPath:          cfg.PythonPath,
		DefaultAPIBase:      resolveModelResearchDefaultAPIBase(cfg),
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

func resolveModelResearchProjectRoot(cfg *config.Config) string {
	candidates := []string{cfg.ConfigDir}
	if cfg.ConfigDir != "" {
		candidates = append(candidates, filepath.Dir(cfg.ConfigDir))
	}
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates, cwd)
	}

	for _, candidate := range candidates {
		if root := findModelResearchProjectRoot(candidate); root != "" {
			return root
		}
	}

	return filepath.Dir(cfg.ConfigDir)
}

func resolveModelResearchDefaultAPIBase(cfg *config.Config) string {
	if envoyURL := strings.TrimRight(strings.TrimSpace(cfg.EnvoyURL), "/"); envoyURL != "" {
		return envoyURL
	}
	return strings.TrimRight(strings.TrimSpace(cfg.RouterAPIURL), "/")
}

func findModelResearchProjectRoot(start string) string {
	current := filepath.Clean(strings.TrimSpace(start))
	if current == "." || current == "" {
		return ""
	}

	for {
		if hasModelResearchScripts(current) {
			return current
		}
		parent := filepath.Dir(current)
		if parent == current {
			return ""
		}
		current = parent
	}
}

func hasModelResearchScripts(root string) bool {
	_, err := os.Stat(filepath.Join(root, "src", "training", "model_eval", "mom_collection_eval.py"))
	return err == nil
}
