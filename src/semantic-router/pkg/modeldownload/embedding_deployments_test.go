package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func findSpecByPath(specs []ModelSpec, localPath string) (ModelSpec, bool) {
	for _, spec := range specs {
		if spec.LocalPath == localPath {
			return spec, true
		}
	}
	return ModelSpec{}, false
}

// Every embedding model is a model_runtime deployment the runtime provisions
// at its pinned revision, so the Router downloads none for its stores or a
// store that names its own deployment.
func TestEmbeddingDeploymentsAreLeftToTheRuntime(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.MoMRegistry = config.ToLegacyRegistry()
	cfg.SemanticCache.InUse = true
	cfg.Memory.InUse = true
	cfg.ModelDeployments["qwen3-embedding"] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "Qwen/Qwen3-Embedding-0.6B", Device: "cpu"}
	cfg.VectorStore = &config.VectorStoreConfig{Enabled: true, BackendType: "memory", Embedding: &config.EmbeddingBinding{Deployment: "qwen3-embedding"}}
	specs, err := BuildModelSpecs(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, found := findSpecByPath(specs, config.VelaEmbeddingModel); found || len(specs) != 0 {
		t.Fatalf("the Router provisions embedding models: %+v", specs)
	}
}
