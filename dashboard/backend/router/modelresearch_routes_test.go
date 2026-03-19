package router

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestResolveModelResearchProjectRootPrefersConfigDirWhenScriptsExistThere(t *testing.T) {
	t.Parallel()

	repoRoot := t.TempDir()
	scriptPath := filepath.Join(repoRoot, "src", "training", "model_eval", "mom_collection_eval.py")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0o755); err != nil {
		t.Fatalf("mkdir scripts: %v", err)
	}
	if err := os.WriteFile(scriptPath, []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("write script: %v", err)
	}

	cfg := &config.Config{ConfigDir: repoRoot}
	if actual := resolveModelResearchProjectRoot(cfg); actual != repoRoot {
		t.Fatalf("resolveModelResearchProjectRoot() = %q, want %q", actual, repoRoot)
	}
}

func TestResolveModelResearchProjectRootWalksUpFromNestedConfigDir(t *testing.T) {
	t.Parallel()

	repoRoot := t.TempDir()
	scriptPath := filepath.Join(repoRoot, "src", "training", "model_eval", "mom_collection_eval.py")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0o755); err != nil {
		t.Fatalf("mkdir scripts: %v", err)
	}
	if err := os.WriteFile(scriptPath, []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("write script: %v", err)
	}

	cfg := &config.Config{ConfigDir: filepath.Join(repoRoot, "config")}
	if actual := resolveModelResearchProjectRoot(cfg); actual != repoRoot {
		t.Fatalf("resolveModelResearchProjectRoot() = %q, want %q", actual, repoRoot)
	}
}

func TestResolveModelResearchDefaultAPIBasePrefersEnvoyURL(t *testing.T) {
	t.Parallel()

	cfg := &config.Config{
		EnvoyURL:     "http://localhost:8899/",
		RouterAPIURL: "http://localhost:8080/",
	}
	if actual := resolveModelResearchDefaultAPIBase(cfg); actual != "http://localhost:8899" {
		t.Fatalf("resolveModelResearchDefaultAPIBase() = %q, want http://localhost:8899", actual)
	}
}

func TestResolveModelResearchDefaultAPIBaseFallsBackToRouterAPIURL(t *testing.T) {
	t.Parallel()

	cfg := &config.Config{RouterAPIURL: "http://localhost:8080/"}
	if actual := resolveModelResearchDefaultAPIBase(cfg); actual != "http://localhost:8080" {
		t.Fatalf("resolveModelResearchDefaultAPIBase() = %q, want http://localhost:8080", actual)
	}
}
