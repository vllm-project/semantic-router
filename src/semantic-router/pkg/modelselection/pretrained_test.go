package modelselection

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

const pretrainedTestModelsEnv = "VLLM_SR_TEST_ML_MODELS_DIR"

// Published model artifacts are an explicit integration-test input. Unit tests
// use testdata and never search for a workspace cache or invoke a downloader.
func pretrainedTestModelsDir() (string, error) {
	dir := os.Getenv(pretrainedTestModelsEnv)
	if dir == "" {
		return "", nil
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(dir)
	if err != nil {
		return "", fmt.Errorf("%s: %w", pretrainedTestModelsEnv, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s must name a directory: %s", pretrainedTestModelsEnv, dir)
	}
	return dir, nil
}

func TestPretrainedArtifactsRequireExplicitDirectory(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	t.Setenv(pretrainedTestModelsEnv, "")
	// A go.mod and old cache beside the working directory must not opt in.
	if err := os.WriteFile("go.mod", []byte("module example.test\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cache := filepath.Join(".cache", "ml-models")
	if err := os.MkdirAll(cache, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cache, "knn_model.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := pretrainedTestModelsDir()
	if err != nil || got != "" {
		t.Fatalf("implicit cache selected: dir=%q err=%v", got, err)
	}
	files, err := os.ReadDir(cache)
	if err != nil || len(files) != 1 {
		t.Fatalf("cache modified: files=%v err=%v", files, err)
	}
}

func TestPretrainedArtifactDirectoryValidation(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(pretrainedTestModelsEnv, dir)
	if got, err := pretrainedTestModelsDir(); err != nil || got != dir {
		t.Fatalf("explicit model directory = %q, %v; want %q", got, err, dir)
	}
	t.Setenv(pretrainedTestModelsEnv, filepath.Join(dir, "missing"))
	if _, err := pretrainedTestModelsDir(); err == nil {
		t.Fatal("missing explicit directory must fail")
	}
	file := filepath.Join(dir, "model.json")
	if err := os.WriteFile(file, []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(pretrainedTestModelsEnv, file)
	if _, err := pretrainedTestModelsDir(); err == nil {
		t.Fatal("file used as explicit directory must fail")
	}
}
