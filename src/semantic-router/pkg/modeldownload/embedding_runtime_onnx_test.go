//go:build onnx

package modeldownload

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildModelSpecsRequiresOnnxEmbeddingArtifacts(t *testing.T) {
	cfg := newEmbeddingOnlyConfig()
	cfg.EmbeddingModels.EmbeddingConfig.TargetLayer = 16

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	spec, ok := findSpecByPath(specs, testEmbeddingModelPath)
	if !ok {
		t.Fatalf("BuildModelSpecs() did not produce %q", testEmbeddingModelPath)
	}

	for _, want := range []string{
		"config.json",
		"tokenizer.json",
		"onnx/layer-22/model.onnx",
		"onnx/layer-22/model.onnx.data",
		"onnx/layer-16/model.onnx",
		"onnx/layer-16/model.onnx.data",
	} {
		if !slices.Contains(spec.RequiredFiles, want) {
			t.Errorf("RequiredFiles = %#v, missing %q", spec.RequiredFiles, want)
		}
	}
	if slices.Contains(spec.RequiredFiles, "model.safetensors") {
		t.Fatalf("ONNX completeness contract requires Candle weights: %#v", spec.RequiredFiles)
	}
	if len(spec.ExcludePatterns) != 0 {
		t.Fatalf("ONNX ExcludePatterns = %#v, want none", spec.ExcludePatterns)
	}
}

func TestBuildModelSpecsRequiresOnnxMultimodalArtifacts(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			testMultiModalModelPath: "llm-semantic-router/multi-modal-embed-small",
		},
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MultiModalModelPath: testMultiModalModelPath,
			},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	spec, ok := findSpecByPath(specs, testMultiModalModelPath)
	if !ok {
		t.Fatalf("BuildModelSpecs() did not produce %q", testMultiModalModelPath)
	}
	for _, want := range append([]string{"config.json"}, onnxMultimodalEmbeddingFiles...) {
		if !slices.Contains(spec.RequiredFiles, want) {
			t.Errorf("RequiredFiles = %#v, missing %q", spec.RequiredFiles, want)
		}
	}
}
