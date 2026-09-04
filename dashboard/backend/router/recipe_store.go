package router

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
)

func newDashboardRecipeStore(cfg *config.Config) *recipe.Store {
	root := strings.TrimSpace(os.Getenv("VLLM_SR_RECIPE_STORE_DIR"))
	if root == "" {
		// ConfigDir may intentionally point at a read-only logical project root.
		// Without an explicit store, keep mutable package state beside the actual
		// runtime configuration, whose directory has already passed the Dashboard
		// writeability probe.
		root = filepath.Join(filepath.Dir(cfg.AbsConfigPath), ".vllm-sr", "recipe-store")
	}
	return recipe.NewStore(recipe.StoreOptions{
		Root:       root,
		ConfigPath: cfg.AbsConfigPath,
	})
}

func dashboardActiveRecipeDirectory(cfg *config.Config) string {
	directory := strings.TrimSpace(os.Getenv("VLLM_SR_ACTIVE_RECIPE_DIR"))
	if directory != "" {
		return directory
	}

	// The Dashboard image ships the latest built-in probe catalog with the CLI.
	// Probes are offline fixtures and remain useful when no mutable Recipe
	// package has been activated, so prefer that catalog over the config root.
	embedded := filepath.Join("/app", "cli", "model_assets", "latest", "mom-v1")
	if info, err := os.Stat(embedded); err == nil && info.IsDir() {
		return embedded
	}
	return cfg.ConfigDir
}

func selectedRecipeStore(cfg *config.Config, stores []*recipe.Store) *recipe.Store {
	if len(stores) > 0 && stores[0] != nil {
		return stores[0]
	}
	return newDashboardRecipeStore(cfg)
}
