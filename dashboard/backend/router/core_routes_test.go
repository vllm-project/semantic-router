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
	if err := os.WriteFile(scriptPath, []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(script): %v", err)
	}
	signalScriptPath := filepath.Join(repoRoot, "src", "training", "model_eval", "signal_eval.py")
	if err := os.WriteFile(signalScriptPath, []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(signal script): %v", err)
	}
	if err := os.MkdirAll(filepath.Join(repoRoot, "dashboard", "backend"), 0o755); err != nil {
		t.Fatalf("MkdirAll(dashboard/backend): %v", err)
	}

	workDir := filepath.Join(repoRoot, "dashboard", "backend")
	previousWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd(): %v", err)
	}
	chdirErr := os.Chdir(workDir)
	if chdirErr != nil {
		t.Fatalf("Chdir(%s): %v", workDir, chdirErr)
	}
	t.Cleanup(func() {
		_ = os.Chdir(previousWD)
	})

	externalConfigDir := filepath.Join(t.TempDir(), "config")
	mkdirErr := os.MkdirAll(externalConfigDir, 0o755)
	if mkdirErr != nil {
		t.Fatalf("MkdirAll(config dir): %v", mkdirErr)
	}

	cfg := &config.Config{ConfigDir: externalConfigDir}
	got := resolveEvaluationProjectRoot(cfg)
	gotResolved, err := filepath.EvalSymlinks(got)
	if err != nil {
		t.Fatalf("EvalSymlinks(got): %v", err)
	}
	wantResolved, err := filepath.EvalSymlinks(repoRoot)
	if err != nil {
		t.Fatalf("EvalSymlinks(repoRoot): %v", err)
	}
	if gotResolved != wantResolved {
		t.Fatalf("resolveEvaluationProjectRoot() = %q, want %q", got, repoRoot)
	}
}

func TestResolveEvaluationProjectRootRecognizesRuntimeAppLayout(t *testing.T) {
	appRoot := t.TempDir()
	scriptDir := filepath.Join(appRoot, "src", "training", "model_eval")
	if err := os.MkdirAll(scriptDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(script dir): %v", err)
	}
	for _, scriptName := range []string{"mmlu_pro_vllm_eval.py", "signal_eval.py"} {
		scriptPath := filepath.Join(scriptDir, scriptName)
		if err := os.WriteFile(scriptPath, []byte("print('ok')\n"), 0o644); err != nil {
			t.Fatalf("WriteFile(%s): %v", scriptName, err)
		}
	}

	previousWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd(): %v", err)
	}
	chdirErr := os.Chdir(appRoot)
	if chdirErr != nil {
		t.Fatalf("Chdir(%s): %v", appRoot, chdirErr)
	}
	t.Cleanup(func() {
		_ = os.Chdir(previousWD)
	})

	cfg := &config.Config{ConfigDir: string(filepath.Separator)}
	got := resolveEvaluationProjectRoot(cfg)
	gotResolved, err := filepath.EvalSymlinks(got)
	if err != nil {
		t.Fatalf("EvalSymlinks(got): %v", err)
	}
	wantResolved, err := filepath.EvalSymlinks(appRoot)
	if err != nil {
		t.Fatalf("EvalSymlinks(appRoot): %v", err)
	}
	if gotResolved != wantResolved {
		t.Fatalf("resolveEvaluationProjectRoot() = %q, want %q", got, appRoot)
	}
}

func TestResolveMLTrainingDirUsesSharedContractPath(t *testing.T) {
	repoRoot := t.TempDir()
	trainingDir := filepath.Join(repoRoot, "src", "training", "model_selection", "ml_model_selection")
	if err := os.MkdirAll(trainingDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(training dir): %v", err)
	}
	benchmarkPath := filepath.Join(trainingDir, "benchmark.py")
	if err := os.WriteFile(benchmarkPath, []byte("print('ok')\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(benchmark): %v", err)
	}

	workDir := filepath.Join(repoRoot, "dashboard", "backend")
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		t.Fatalf("MkdirAll(dashboard/backend): %v", err)
	}

	previousWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd(): %v", err)
	}
	chdirErr := os.Chdir(workDir)
	if chdirErr != nil {
		t.Fatalf("Chdir(%s): %v", workDir, chdirErr)
	}
	t.Cleanup(func() {
		_ = os.Chdir(previousWD)
	})

	externalConfigDir := filepath.Join(t.TempDir(), "config")
	mkdirErr := os.MkdirAll(externalConfigDir, 0o755)
	if mkdirErr != nil {
		t.Fatalf("MkdirAll(config dir): %v", mkdirErr)
	}

	cfg := &config.Config{ConfigDir: externalConfigDir}
	got := resolveMLTrainingDir(cfg)
	gotResolved, err := filepath.EvalSymlinks(got)
	if err != nil {
		t.Fatalf("EvalSymlinks(got): %v", err)
	}
	wantResolved, err := filepath.EvalSymlinks(trainingDir)
	if err != nil {
		t.Fatalf("EvalSymlinks(trainingDir): %v", err)
	}
	if gotResolved != wantResolved {
		t.Fatalf("resolveMLTrainingDir() = %q, want %q", got, trainingDir)
	}
}
