package modeldownload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Tests for the candle-embedding-runtime completeness requirements
// (collectCandleEmbeddingRequiredFiles and its wiring through
// ExtractRequiredFilesByModel / BuildModelSpecs).

func TestExtractRequiredFilesByModelIncludesCandleEmbeddingRuntimeFiles(t *testing.T) {
	tests := []struct {
		name      string
		backend   string
		modelType string
		want      bool
	}{
		{name: "empty backend defaults to candle", backend: "", want: true},
		{name: "explicit candle backend", backend: "candle", want: true},
		{name: "candle backend is case and space insensitive", backend: "  Candle ", want: true},
		{name: "openvino backend adds no candle requirements", backend: "openvino", want: false},
		{name: "openai_compatible backend adds no candle requirements", backend: "openai_compatible", want: false},
		{name: "remote model_type with empty backend resolves remote, no candle requirements", backend: "", modelType: "remote", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.RouterConfig{
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{
						Qwen3ModelPath:  "models/mom-embedding-pro",
						MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
						EmbeddingConfig: config.HNSWConfig{Backend: tt.backend, ModelType: tt.modelType},
					},
				},
			}

			got := ExtractRequiredFilesByModel(cfg)
			for _, modelPath := range []string{
				"models/mom-embedding-pro",
				"models/mmbert-embed-32k-2d-matryoshka",
			} {
				for _, file := range []string{"model.safetensors", "tokenizer.json"} {
					if slices.Contains(got[modelPath], file) != tt.want {
						t.Errorf("ExtractRequiredFilesByModel()[%q] contains %q = %v, want %v (map: %#v)",
							modelPath, file, !tt.want, tt.want, got)
					}
				}
			}
		})
	}
}

func TestExtractRequiredFilesByModelSkipsNonLocalEmbeddingPaths(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "/opt/external/mmbert",
			},
		},
	}

	got := ExtractRequiredFilesByModel(cfg)
	if _, ok := got["/opt/external/mmbert"]; ok {
		t.Fatalf("ExtractRequiredFilesByModel() recorded requirements for non-models/ path: %#v", got)
	}
}

func TestExtractRequiredFilesByModelCoversMultiModalIndependentOfBackend(t *testing.T) {
	// The multimodal initializer branch runs before the backend switch
	// whenever model_type resolves to "multimodal", so its loader files are
	// required regardless of the configured backend.
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath:     "models/mmbert-embed-32k-2d-matryoshka",
				MultiModalModelPath: "models/mom-embedding-multimodal",
				EmbeddingConfig: config.HNSWConfig{
					Backend:   "openvino",
					ModelType: "multimodal",
				},
			},
		},
	}

	got := ExtractRequiredFilesByModel(cfg)
	for _, file := range []string{"model.safetensors", "tokenizer.json"} {
		if !slices.Contains(got["models/mom-embedding-multimodal"], file) {
			t.Errorf("multimodal path missing required file %q under openvino backend (map: %#v)", file, got)
		}
		if slices.Contains(got["models/mmbert-embed-32k-2d-matryoshka"], file) {
			t.Errorf("mmbert path gained candle requirement %q under openvino backend (map: %#v)", file, got)
		}
	}

	// Without model_type=multimodal the multimodal path gains no requirement.
	cfg.EmbeddingModels.EmbeddingConfig.ModelType = ""
	got = ExtractRequiredFilesByModel(cfg)
	if _, ok := got["models/mom-embedding-multimodal"]; ok {
		t.Errorf("multimodal path recorded requirements without model_type=multimodal: %#v", got)
	}
}

// Regression test for the partial-download poisoning scenario: an interrupted
// hf download leaves companion artifacts (config.json, nested onnx/ exports)
// without the root weight files the candle embedding runtime hard-loads.
// The nested onnx file satisfies the generic weight heuristic, so without the
// candle-derived RequiredFiles the model reads as complete, is never
// re-downloaded, and the router crash-loops at classifier init on every
// restart ("failed to load safetensors from .../model.safetensors").
func TestPartiallyDownloadedCandleEmbeddingModelIsIncomplete(t *testing.T) {
	tmpDir := t.TempDir()
	modelDir := writeModelDir(t, tmpDir, "mmbert-partial", map[string]string{
		"config.json":                     "{}",
		"onnx/layer-6/model_fa_fp16.onnx": "",
	})

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
			},
		},
	}
	requiredFiles := append([]string{}, DefaultRequiredFiles...)
	requiredFiles = append(requiredFiles,
		ExtractRequiredFilesByModel(cfg)["models/mmbert-embed-32k-2d-matryoshka"]...)

	complete, err := IsModelComplete(modelDir, requiredFiles)
	if err != nil {
		t.Fatalf("IsModelComplete() error = %v", err)
	}
	if complete {
		t.Fatal("IsModelComplete() = true for a partial download missing model.safetensors and tokenizer.json, want false")
	}

	// Once the runtime-required files land, the same model reads as complete.
	for _, file := range []string{"model.safetensors", "tokenizer.json"} {
		if writeErr := os.WriteFile(filepath.Join(modelDir, file), []byte(""), 0o644); writeErr != nil {
			t.Fatalf("WriteFile(%q) failed: %v", file, writeErr)
		}
	}
	complete, err = IsModelComplete(modelDir, requiredFiles)
	if err != nil {
		t.Fatalf("IsModelComplete() error = %v", err)
	}
	if !complete {
		t.Fatal("IsModelComplete() = false after weights and tokenizer are present, want true")
	}
}

func TestExtractRequiredFilesByModelGemmaIncludesDenseBottleneckWeights(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				GemmaModelPath:  "models/mom-embedding-flash",
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
			},
		},
	}

	got := ExtractRequiredFilesByModel(cfg)
	for _, file := range []string{"model.safetensors", "tokenizer.json", "2_Dense/model.safetensors", "3_Dense/model.safetensors"} {
		if !slices.Contains(got["models/mom-embedding-flash"], file) {
			t.Errorf("gemma path missing required file %q (map: %#v)", file, got)
		}
	}
	for _, file := range []string{"2_Dense/model.safetensors", "3_Dense/model.safetensors"} {
		if slices.Contains(got["models/mmbert-embed-32k-2d-matryoshka"], file) {
			t.Errorf("mmbert path gained gemma-only dense requirement %q (map: %#v)", file, got)
		}
	}
}

func TestExtractRequiredFilesByModelCoversMultiModalForImageCandidates(t *testing.T) {
	// Complexity rules with image_candidates load the multimodal model with
	// no model_type gate, so the requirement must follow without it.
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MultiModalModelPath: "models/mom-embedding-multimodal",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ComplexityRules: []config.ComplexityRule{
					{Hard: config.ComplexityCandidates{ImageCandidates: []string{"a screenshot of code"}}},
				},
			},
		},
	}

	got := ExtractRequiredFilesByModel(cfg)
	for _, file := range []string{"model.safetensors", "tokenizer.json"} {
		if !slices.Contains(got["models/mom-embedding-multimodal"], file) {
			t.Errorf("multimodal path missing %q under image_candidates trigger (map: %#v)", file, got)
		}
	}
}

func TestExtractRequiredFilesByModelMultiModalEdgeCases(t *testing.T) {
	// model_type matching is case- and space-insensitive, mirroring the
	// classification initializer; an empty multimodal path records nothing.
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MultiModalModelPath: "models/mom-embedding-multimodal",
				EmbeddingConfig:     config.HNSWConfig{ModelType: "  MultiModal "},
			},
		},
	}
	got := ExtractRequiredFilesByModel(cfg)
	if !slices.Contains(got["models/mom-embedding-multimodal"], "model.safetensors") {
		t.Errorf("case/space-insensitive model_type did not trigger multimodal requirement (map: %#v)", got)
	}

	cfg.EmbeddingModels.MultiModalModelPath = ""
	got = ExtractRequiredFilesByModel(cfg)
	if len(got) != 0 {
		t.Errorf("empty multimodal path recorded requirements: %#v", got)
	}
}

func TestBuildModelSpecsIncludesCandleEmbeddingRequiredFiles(t *testing.T) {
	// End-to-end through BuildModelSpecs: the production path must carry the
	// candle runtime files into the spec used by GetMissingModels.
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mmbert-embed-32k-2d-matryoshka",
			},
		},
		MoMRegistry: map[string]string{
			"models/mmbert-embed-32k-2d-matryoshka": "llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 1 (%#v)", len(specs), specs)
	}
	for _, file := range []string{"config.json", "model.safetensors", "tokenizer.json"} {
		if !slices.Contains(specs[0].RequiredFiles, file) {
			t.Errorf("spec.RequiredFiles missing %q (got %#v)", file, specs[0].RequiredFiles)
		}
	}
}
