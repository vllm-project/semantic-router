package router

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestResolveEvaluationProjectRootFallsBackToWorkingDirectoryRepo(t *testing.T) {
	repoRoot := t.TempDir()
	scriptPath := filepath.Join(repoRoot, "src", "training", "model_eval", "mmlu_pro_vllm_eval.py")
	if err := os.MkdirAll(filepath.Dir(scriptPath), 0o755); err != nil {
		t.Fatalf("MkdirAll(script dir): %v", err)
	}
	if err := os.MkdirAll(filepath.Join(repoRoot, "dashboard", "backend"), 0o755); err != nil {
		t.Fatalf("MkdirAll(dashboard/backend): %v", err)
	}
	if err := os.WriteFile(filepath.Join(repoRoot, "AGENTS.md"), []byte("test"), 0o644); err != nil {
		t.Fatalf("WriteFile(AGENTS.md): %v", err)
	}
	if err := os.WriteFile(scriptPath, []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(script): %v", err)
	}

	workDir := filepath.Join(repoRoot, "dashboard", "backend")
	previousWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd(): %v", err)
	}
	if err := os.Chdir(workDir); err != nil {
		t.Fatalf("Chdir(%s): %v", workDir, err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(previousWD)
	})

	externalConfigDir := filepath.Join(t.TempDir(), "config")
	if err := os.MkdirAll(externalConfigDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(config dir): %v", err)
	}

	cfg := &config.Config{ConfigDir: externalConfigDir}
	if got := resolveEvaluationProjectRoot(cfg); got != repoRoot {
		t.Fatalf("resolveEvaluationProjectRoot() = %q, want %q", got, repoRoot)
	}
}
