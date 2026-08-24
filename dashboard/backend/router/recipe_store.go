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
		root = filepath.Join(cfg.ConfigDir, ".vllm-sr", "recipe-store")
	}
	return recipe.NewStore(recipe.StoreOptions{
		Root:       root,
		ConfigPath: cfg.AbsConfigPath,
	})
}

func dashboardActiveRecipeDirectory(cfg *config.Config) string {
	directory := strings.TrimSpace(os.Getenv("VLLM_SR_ACTIVE_RECIPE_DIR"))
	if directory == "" {
		return cfg.ConfigDir
	}
	return directory
}

func selectedRecipeStore(cfg *config.Config, stores []*recipe.Store) *recipe.Store {
	if len(stores) > 0 && stores[0] != nil {
		return stores[0]
	}
	return newDashboardRecipeStore(cfg)
}
