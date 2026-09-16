//go:build !windows && cgo && (amd64 || arm64)

package modelruntime

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

func TestGlobalEmbeddingViewDoesNotChangeOtherModelFamilies(t *testing.T) {
	provider, _ := config.DefaultModelExecution(true)
	if provider != "ort" || os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires ORT build-default and real ONNX Runtime")
	}
	fixture, pathErr := filepath.Abs(filepath.Join("..", "..", "..", "..", "onnx-binding", "instance", "testdata", "embedding"))
	if pathErr != nil {
		t.Fatal(pathErr)
	}
	primary := t.TempDir()
	for _, name := range []string{"config.json", "tokenizer.json", "onnx/layer-1/model.onnx", "onnx/layer-2/model.onnx"} {
		data, readErr := os.ReadFile(filepath.Join(fixture, filepath.Base(name)))
		if readErr != nil {
			t.Fatal(readErr)
		}
		target := filepath.Join(primary, name)
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(primary, "onnx", "model_config.json"), []byte(`{"available_layers":[1,2]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.EmbeddingModels.UseCPU = true
	cfg.EmbeddingConfig.ModelType, cfg.EmbeddingConfig.TargetLayer, cfg.EmbeddingConfig.TargetDimension = "mmbert", 2, 3
	cfg.Qwen3ModelPath = fixture // This complete graph has no layer-2 view.
	cfg.Tools.Enabled = true
	cfg.ModelSelection.ML.ModelsPath, cfg.ModelSelection.ML.ModelType = "selectors", "qwen3"
	cfg.Decisions = []config.Decision{{Algorithm: &config.AlgorithmConfig{Type: "knn"}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"primary": {Artifact: primary, Provider: "ort", Device: "cpu", Input: config.ModelInputBudget{MaxTokens: 8, Overflow: "reject"}}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "primary", Contract: "embedding.v1", Adapter: "mmbert", Head: "onnx/layer-2/model.onnx"}}
	runtime := native.New(nil)
	prepared, err := PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	for _, family := range []string{"mmbert", "qwen3"} {
		provider, err := prepared.Get(family, 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		vector, err := provider.Embed(context.Background(), "hello world")
		if err != nil || len(vector) != 3 {
			t.Fatalf("%s view contaminated: %v / %v", family, vector, err)
		}
	}
}
