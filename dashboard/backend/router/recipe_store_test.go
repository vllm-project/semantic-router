package router

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestDashboardRecipeStoreDefaultsBesideRuntimeConfig(t *testing.T) {
	t.Setenv("VLLM_SR_RECIPE_STORE_DIR", "")
	runtimeDirectory := t.TempDir()
	store := newDashboardRecipeStore(&config.Config{
		AbsConfigPath: filepath.Join(runtimeDirectory, "config.yaml"),
		ConfigDir:     filepath.Join(t.TempDir(), "logical-root"),
	})

	if got, want := store.Root(), filepath.Join(runtimeDirectory, ".vllm-sr", "recipe-store"); got != want {
		t.Fatalf("store root = %q, want %q", got, want)
	}
}

func TestDashboardActiveRecipeDirectoryPrefersExplicitCatalog(t *testing.T) {
	directory := t.TempDir()
	t.Setenv("VLLM_SR_ACTIVE_RECIPE_DIR", directory)

	if got := dashboardActiveRecipeDirectory(&config.Config{ConfigDir: t.TempDir()}); got != directory {
		t.Fatalf("active Recipe directory = %q, want %q", got, directory)
	}
}

func TestDashboardActiveRecipeDirectoryFallsBackToConfigDirectory(t *testing.T) {
	t.Setenv("VLLM_SR_ACTIVE_RECIPE_DIR", "")
	configDirectory := t.TempDir()
	// Unit tests run outside the Dashboard image. Keep this assertion explicit
	// so a developer machine with an unrelated /app tree cannot change it.
	if info, err := os.Stat(filepath.Join("/app", "cli", "model_assets", "latest", "mom-v1")); err == nil && info.IsDir() {
		t.Skip("Dashboard image built-in catalog is present")
	}

	if got := dashboardActiveRecipeDirectory(&config.Config{ConfigDir: configDirectory}); got != configDirectory {
		t.Fatalf("active Recipe directory = %q, want %q", got, configDirectory)
	}
}
