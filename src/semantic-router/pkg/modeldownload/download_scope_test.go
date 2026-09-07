package modeldownload

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestBuildModelSpecsExcludesOnnxWeightsForCandleEmbeddingModels guards the download
// scope for the default local backend: the candle runtime loads model.safetensors +
// tokenizer.json, so the multi-gigabyte ONNX exports shipped in the same repository
// must not be fetched. Every candle embedding path gets the same narrowing.
func TestBuildModelSpecsExcludesOnnxWeightsForCandleEmbeddingModels(t *testing.T) {
	specs, err := BuildModelSpecs(newCandleEmbeddingConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	for _, modelPath := range []string{
		testEmbeddingModelPath,
		testQwen3ModelPath,
		testGemmaModelPath,
		testMultiModalModelPath,
	} {
		spec, ok := findSpecByPath(specs, modelPath)
		if !ok {
			t.Fatalf("BuildModelSpecs() did not produce a spec for %q", modelPath)
		}
		if !reflect.DeepEqual(spec.ExcludePatterns, onnxWeightExcludePatterns) {
			t.Fatalf("%s ExcludePatterns = %#v, want %#v", modelPath, spec.ExcludePatterns, onnxWeightExcludePatterns)
		}
	}
}

// TestBuildModelSpecsExcludesOnnxWeightsForAliasedEmbeddingModel keeps the narrowing
// attached to the model when the config names it by a registry alias. The exclude map
// is keyed and looked up by the canonical path, so it must match whether the collected
// provisioning path is the literal alias or has already been canonicalized (#2828).
func TestBuildModelSpecsExcludesOnnxWeightsForAliasedEmbeddingModel(t *testing.T) {
	for _, configured := range []string{
		"models/mom-embedding-ultra", // models/-prefixed alias
		testEmbeddingModelPath,       // canonical path
	} {
		t.Run(configured, func(t *testing.T) {
			cfg := &config.RouterConfig{
				MoMRegistry: config.ToLegacyRegistry(),
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{MmBertModelPath: configured},
				},
			}

			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatalf("BuildModelSpecs() error = %v", err)
			}

			found := false
			for _, spec := range specs {
				if config.ResolveModelPath(spec.LocalPath) != testEmbeddingModelPath {
					continue
				}
				found = true
				if !reflect.DeepEqual(spec.ExcludePatterns, onnxWeightExcludePatterns) {
					t.Fatalf("%s ExcludePatterns = %#v, want %#v", spec.LocalPath, spec.ExcludePatterns, onnxWeightExcludePatterns)
				}
			}
			if !found {
				t.Fatalf("BuildModelSpecs() produced no spec resolving to %q; got %#v", testEmbeddingModelPath, specs)
			}
		})
	}
}

// TestBuildModelSpecsKeepsFullSnapshotForOpenVINOBackend keeps ONNX deployments whole:
// the OpenVINO embedding backend consumes the ONNX exports, so it must keep receiving
// the unfiltered repository.
func TestBuildModelSpecsKeepsFullSnapshotForOpenVINOBackend(t *testing.T) {
	cfg := newCandleEmbeddingConfig()
	cfg.EmbeddingModels.EmbeddingConfig = config.HNSWConfig{
		Backend: config.EmbeddingBackendOpenVINO,
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	for _, spec := range specs {
		if len(spec.ExcludePatterns) != 0 {
			t.Fatalf("%s ExcludePatterns = %#v, want none for the openvino backend", spec.LocalPath, spec.ExcludePatterns)
		}
	}
}

// TestBuildModelSpecsLeavesNonEmbeddingModelsUnfiltered limits the blast radius to the
// embedding runtime: other locally provisioned models keep the full snapshot until their
// own runtime contract is encoded.
func TestBuildModelSpecsLeavesNonEmbeddingModelsUnfiltered(t *testing.T) {
	const bertModelPath = "models/all-MiniLM-L12-v2"
	cfg := newEmbeddingOnlyConfig()
	cfg.MoMRegistry[bertModelPath] = "sentence-transformers/all-MiniLM-L12-v2"
	cfg.BertModelPath = bertModelPath

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	spec, ok := findSpecByPath(specs, bertModelPath)
	if !ok {
		t.Fatalf("BuildModelSpecs() did not produce a spec for %q; got %#v", bertModelPath, specs)
	}
	if len(spec.ExcludePatterns) != 0 {
		t.Fatalf("%s ExcludePatterns = %#v, want none", bertModelPath, spec.ExcludePatterns)
	}
}

// TestOnnxWeightExcludePatternsNeverMatchCandleRequiredFiles keeps the exclude list
// and the completeness contract aligned: a pattern that matched a hard-loaded file
// would make every download incomplete and loop forever.
func TestOnnxWeightExcludePatternsNeverMatchCandleRequiredFiles(t *testing.T) {
	required := candleEmbeddingModelRequiredFiles(newCandleEmbeddingConfig())
	protected := append([]string{}, DefaultRequiredFiles...)
	protected = append(protected, "onnx/model_config.json")
	for _, files := range required {
		protected = append(protected, files...)
	}

	for _, pattern := range onnxWeightExcludePatterns {
		suffix := strings.TrimPrefix(pattern, "*")
		if suffix == pattern {
			t.Fatalf("exclude pattern %q must be a suffix glob so it cannot shadow required files", pattern)
		}
		for _, file := range protected {
			if strings.HasSuffix(file, suffix) {
				t.Fatalf("exclude pattern %q matches required file %q", pattern, file)
			}
		}
	}
}
