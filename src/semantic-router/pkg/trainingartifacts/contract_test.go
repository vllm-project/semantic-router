package trainingartifacts

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCurrentContractIncludesSharedTrainingArtifacts(t *testing.T) {
	t.Parallel()

	contract := CurrentContract()
	if contract.Version == "" {
		t.Fatal("expected contract version")
	}
	if contract.ModelEval.RepoRelativeDir != "src/training/model_eval" {
		t.Fatalf("ModelEval.RepoRelativeDir = %q", contract.ModelEval.RepoRelativeDir)
	}
	if contract.MLPipeline.RepoRelativeDir != "src/training/model_selection/ml_model_selection" {
		t.Fatalf("MLPipeline.RepoRelativeDir = %q", contract.MLPipeline.RepoRelativeDir)
	}
	if contract.ModelEval.Outputs.DefaultConfigOutputFile != "config/config.eval.yaml" {
		t.Fatalf("DefaultConfigOutputFile = %q", contract.ModelEval.Outputs.DefaultConfigOutputFile)
	}
	if contract.MLPipeline.Outputs.BenchmarkOutputFile != "benchmark_output.jsonl" {
		t.Fatalf("BenchmarkOutputFile = %q", contract.MLPipeline.Outputs.BenchmarkOutputFile)
	}
	if contract.RuntimeDefaults.CategoryReasoning["math"] != true {
		t.Fatal("expected math category reasoning to be enabled")
	}
}

func TestFindProjectRootWithTrainingLayouts(t *testing.T) {
	t.Parallel()

	repoRoot := t.TempDir()
	evalDir := filepath.Join(repoRoot, "src", "training", "model_eval")
	writeTestFile(t, filepath.Join(evalDir, "signal_eval.py"))
	writeTestFile(t, filepath.Join(evalDir, "mmlu_pro_vllm_eval.py"))
	mlDir := filepath.Join(repoRoot, "src", "training", "model_selection", "ml_model_selection")
	writeTestFile(t, filepath.Join(mlDir, "benchmark.py"))

	nested := filepath.Join(repoRoot, "dashboard", "backend")
	writeTestDir(t, nested)

	if got := FindProjectRootWithModelEval(nested); got != repoRoot {
		t.Fatalf("FindProjectRootWithModelEval() = %q, want %q", got, repoRoot)
	}
	if got := FindProjectRootWithMLPipeline(nested); got != repoRoot {
		t.Fatalf("FindProjectRootWithMLPipeline() = %q, want %q", got, repoRoot)
	}
}

func TestContractHelperPaths(t *testing.T) {
	t.Parallel()

	contract := CurrentContract()
	projectRoot := "/repo"

	signalEvalPath, err := contract.ModelEvalScriptPath(projectRoot, ModelEvalSignalEvalScript)
	if err != nil {
		t.Fatalf("ModelEvalScriptPath(signal_eval): %v", err)
	}
	if signalEvalPath != filepath.Join("/repo", "src", "training", "model_eval", "signal_eval.py") {
		t.Fatalf("signalEvalPath = %q", signalEvalPath)
	}

	if got := contract.SignalEvalOutputPath("/tmp/results", "mmlu-pro-en"); got != filepath.Join("/tmp/results", "signal_eval_mmlu-pro-en.json") {
		t.Fatalf("SignalEvalOutputPath() = %q", got)
	}
	if got := contract.TrainOutputDir("/tmp/data"); got != filepath.Join("/tmp/data", "ml-train") {
		t.Fatalf("TrainOutputDir() = %q", got)
	}

	modelPath, err := contract.ModelArtifactPath("/tmp/models", "svm")
	if err != nil {
		t.Fatalf("ModelArtifactPath(svm): %v", err)
	}
	if modelPath != filepath.Join("/tmp/models", "svm_model.json") {
		t.Fatalf("ModelArtifactPath() = %q", modelPath)
	}
}

func writeTestFile(t *testing.T, path string) {
	t.Helper()
	writeTestDir(t, filepath.Dir(path))
	if err := os.WriteFile(path, []byte("ok\n"), 0o644); err != nil {
		t.Fatalf("WriteFile(%s): %v", path, err)
	}
}

func writeTestDir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("MkdirAll(%s): %v", path, err)
	}
}
