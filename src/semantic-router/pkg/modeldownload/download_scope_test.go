package modeldownload

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestBuildModelSpecsExcludesOnnxWeightsForLocalEmbeddingModels guards the download
// scope for local embeddings: the model runtime loads model.safetensors +
// tokenizer.json, so the multi-gigabyte ONNX exports shipped in the same repository
// must not be fetched. Every local embedding path gets the same narrowing.
func TestBuildModelSpecsExcludesOnnxWeightsForLocalEmbeddingModels(t *testing.T) {
	specs, err := BuildModelSpecs(newLocalEmbeddingConfig())
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	for _, modelPath := range []string{
		testEmbeddingModelPath,
		testQwen3ModelPath,
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

			cfg.EmbeddingConfig.ModelType = "mmbert"
			cfg.Tools.Enabled = true
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

// TestEmbeddingExcludePatternsLeaveOtherModelsUnfiltered limits the blast radius to the
// embedding runtime: other locally provisioned models keep the full snapshot until their
// own runtime contract is encoded.
func TestEmbeddingExcludePatternsLeaveOtherModelsUnfiltered(t *testing.T) {
	cfg := newEmbeddingOnlyConfig()
	cfg.CategoryModel.ModelID = "models/mom-domain-classifier"
	excluded := embeddingModelExcludePatterns(cfg)
	if patterns, ok := excluded["models/mom-domain-classifier"]; ok {
		t.Fatalf("a classifier snapshot is filtered: %#v", patterns)
	}
	if got := excluded[config.ResolveModelPath(testEmbeddingModelPath)]; !reflect.DeepEqual(got, onnxWeightExcludePatterns) {
		t.Fatalf("embedding ExcludePatterns = %#v", got)
	}
}

// TestOnnxWeightExcludePatternsNeverMatchRuntimeRequiredFiles keeps the exclude list
// and the completeness contract aligned: a pattern that matched a hard-loaded file
// would make every download incomplete and loop forever.
func TestOnnxWeightExcludePatternsNeverMatchRuntimeRequiredFiles(t *testing.T) {
	required := embeddingModelRequiredFiles(newLocalEmbeddingConfig())
	protected := append([]string{}, DefaultRequiredFiles...)
	protected = append(protected, "onnx/model_config.json")
	for _, files := range required {
		protected = append(protected, files...)
	}

	for _, pattern := range onnxWeightExcludePatterns {
		for _, file := range protected {
			if revisionArtifactExcluded(file, []string{pattern}) {
				t.Fatalf("exclude pattern %q matches required file %q", pattern, file)
			}
		}
	}
}
