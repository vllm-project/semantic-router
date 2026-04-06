package mlpipeline

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingartifacts"
)

func TestTrainDirUsesSharedArtifactContract(t *testing.T) {
	t.Parallel()

	dataDir := t.TempDir()
	runner := NewRunner(RunnerConfig{DataDir: dataDir})
	want := filepath.Join(dataDir, trainingartifacts.CurrentContract().MLPipeline.Outputs.TrainOutputDirName)
	if got := runner.TrainDir(); got != want {
		t.Fatalf("TrainDir() = %q, want %q", got, want)
	}
}

func TestBuildBenchmarkArgsUsesSharedBenchmarkScriptName(t *testing.T) {
	t.Parallel()

	runner := &Runner{trainingDir: "/tmp/training"}
	args, concurrency := runner.buildBenchmarkArgs(
		BenchmarkRequest{},
		"/tmp/models.yaml",
		"/tmp/queries.jsonl",
		"/tmp/output.jsonl",
	)

	if concurrency != 4 {
		t.Fatalf("concurrency = %d, want 4", concurrency)
	}
	wantScript := filepath.Join("/tmp/training", trainingartifacts.CurrentContract().MLPipeline.Scripts.Benchmark)
	if args[0] != wantScript {
		t.Fatalf("args[0] = %q, want %q", args[0], wantScript)
	}
}

func TestBuildConfigYAMLUsesSharedArtifactFileNames(t *testing.T) {
	t.Parallel()

	contract := trainingartifacts.CurrentContract()
	config := buildConfigYAML(
		ConfigRequest{
			Device: "cuda",
			Decisions: []DecisionEntry{
				{Name: "knn", Algorithm: "knn"},
				{Name: "mlp", Algorithm: "mlp"},
			},
		},
	)

	modelsPath := filepath.Join("/data/ml-pipeline", contract.MLPipeline.Outputs.TrainOutputDirName)
	if config.Config.ModelSelection.ML.ModelsPath != modelsPath {
		t.Fatalf("ModelsPath = %q, want %q", config.Config.ModelSelection.ML.ModelsPath, modelsPath)
	}
	if config.Config.ModelSelection.ML.KNN == nil {
		t.Fatal("expected KNN config")
	}
	if config.Config.ModelSelection.ML.KNN.PretrainedPath != filepath.Join(modelsPath, contract.MLPipeline.Outputs.ModelFiles.KNN) {
		t.Fatalf("KNN.PretrainedPath = %q", config.Config.ModelSelection.ML.KNN.PretrainedPath)
	}
	if config.Config.ModelSelection.ML.MLP == nil {
		t.Fatal("expected MLP config")
	}
	if config.Config.ModelSelection.ML.MLP.PretrainedPath != filepath.Join(modelsPath, contract.MLPipeline.Outputs.ModelFiles.MLP) {
		t.Fatalf("MLP.PretrainedPath = %q", config.Config.ModelSelection.ML.MLP.PretrainedPath)
	}
}
