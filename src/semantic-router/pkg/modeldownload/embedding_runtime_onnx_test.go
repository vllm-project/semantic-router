package modeldownload

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestExplicitORTEmbeddingRequiresSelectedLayerInCandleProcess(t *testing.T) {
	cfg := newEmbeddingOnlyConfig()
	cfg.EmbeddingConfig.TargetLayer = 16
	cfg.ModelDeployments = map[string]config.ModelDeployment{"embed": {Provider: "ort", Artifact: testEmbeddingModelPath}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "embed", Contract: "embedding.v1", Adapter: "mmbert"}}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, ok := findSpecByPath(specs, testEmbeddingModelPath)
	if !ok {
		t.Fatal("embedding artifact missing")
	}
	if !slices.Contains(spec.RequiredFiles, "onnx/layer-16/model.onnx") || !spec.CheckONNX || len(spec.ExcludePatterns) != 0 {
		t.Fatalf("ORT required files=%#v", spec)
	}
	if slices.Contains(spec.RequiredFiles, "model.safetensors") {
		t.Fatal("explicit ORT requires Candle weights")
	}
}

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
