package modeldownload

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestUnusedEmbeddingCatalogEntriesAreNotDownloaded(t *testing.T) {
	cfg := newEmbeddingOnlyConfig()
	cfg.Tools.Enabled = false
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 0 {
		t.Fatalf("unused catalog entries downloaded: %#v", specs)
	}
}

// The runtime's native engine loads safetensors weights on every device: an
// implicit embedding model downloads them without the ONNX exports, and an
// explicit model_runtime deployment is the runtime's own download.
func TestEmbeddingProvisioningFollowsTheRuntime(t *testing.T) {
	cfg := newEmbeddingOnlyConfig()
	cfg.EmbeddingConfig.TargetLayer = 6
	cfg.MmBertModelPath = "models/mom-embedding-ultra"
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, ok := findSpecByPath(specs, testEmbeddingModelPath)
	if !ok {
		t.Fatal("implicit aliased embedding artifact missing")
	}
	if !slices.Contains(spec.RequiredFiles, "model.safetensors") || !slices.Contains(spec.ExcludePatterns, "*.onnx") || spec.CheckONNX {
		t.Fatalf("implicit provisioning does not require native weights: %#v", spec)
	}

	cfg.ModelDeployments = map[string]config.ModelDeployment{"explicit": {Provider: config.ModelRuntimeProvider, Artifact: testEmbeddingModelPath}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "explicit", Contract: "embedding.v1", Adapter: "mmbert"}}
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := findSpecByPath(specs, testEmbeddingModelPath); ok {
		t.Fatalf("the router downloaded a runtime-served embedding model: %#v", specs)
	}
}
