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

// Run with the ROCm image's build default as well as the regular CPU build.
// Provisioning must require the format that implicit runtime preparation opens.
func TestImplicitEmbeddingProvisioningFollowsBuildProvider(t *testing.T) {
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
	provider, _ := config.DefaultModelExecution(true)
	if provider == "ort" {
		if !spec.CheckONNX || !slices.Contains(spec.RequiredFiles, "onnx/layer-6/model.onnx") || len(spec.ExcludePatterns) != 0 {
			t.Fatalf("implicit ORT provisioning does not include its graph: %#v", spec)
		}
		if slices.Contains(spec.RequiredFiles, "model.safetensors") {
			t.Fatal("implicit ORT unexpectedly requires Candle weights")
		}
	} else if !slices.Contains(spec.RequiredFiles, "model.safetensors") || !slices.Contains(spec.ExcludePatterns, "*.onnx") {
		t.Fatalf("implicit Candle provisioning does not require its weights: %#v", spec)
	}

	cfg.ModelDeployments = map[string]config.ModelDeployment{"explicit": {Provider: "candle", Artifact: testEmbeddingModelPath}}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "explicit", Contract: "embedding.v1", Adapter: "mmbert"}}
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, ok = findSpecByPath(specs, testEmbeddingModelPath)
	if !ok || spec.CheckONNX || !slices.Contains(spec.ExcludePatterns, "*.onnx") {
		t.Fatalf("explicit Candle binding did not override build default: %#v", spec)
	}
}
