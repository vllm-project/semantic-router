package modeldownload

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestBuildModelSpecsTargetsTheDirectoryTheRuntimeLoads guards #2676. Every consumer of an
// embedding model path canonicalizes it with config.ResolveModelPath before handing it to
// candle, so a config that names the model by a registry alias must be downloaded to the
// canonical directory. Recording the literal path instead means a bare alias is never
// downloaded at all, and a models/<alias> form is downloaded to a directory nobody opens.
func TestBuildModelSpecsTargetsTheDirectoryTheRuntimeLoads(t *testing.T) {
	tests := []struct {
		name       string
		configured string
	}{
		{name: "canonical path", configured: "models/mmbert-embed-32k-2d-matryoshka"},
		{name: "bare alias", configured: "mmbert"},
		{name: "models-prefixed alias", configured: "models/mom-embedding-ultra"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.RouterConfig{
				MoMRegistry: config.ToLegacyRegistry(),
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{MmBertModelPath: tt.configured},
				},
			}

			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatalf("BuildModelSpecs() error = %v", err)
			}

			loadedByRuntime := config.ResolveModelPath(tt.configured)
			if len(specs) != 1 {
				t.Fatalf("BuildModelSpecs() returned %d specs for %q, want 1 for %q: %#v",
					len(specs), tt.configured, loadedByRuntime, specs)
			}
			if specs[0].LocalPath != loadedByRuntime {
				t.Fatalf("BuildModelSpecs() downloads %q, but the runtime loads %q",
					specs[0].LocalPath, loadedByRuntime)
			}
			if want := "llm-semantic-router/mmbert-embed-32k-2d-matryoshka"; specs[0].RepoID != want {
				t.Fatalf("spec RepoID = %q, want %q", specs[0].RepoID, want)
			}
		})
	}
}

// TestAliasedEmbeddingModelKeepsWeightRequirements pins that the #2172 completeness contract
// follows the model to its canonical directory. Requirements keyed by the alias would leave
// the downloaded snapshot with only the generic weight heuristic behind it.
func TestAliasedEmbeddingModelKeepsWeightRequirements(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: config.ToLegacyRegistry(),
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{MmBertModelPath: "models/mom-embedding-ultra"},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	spec, ok := findSpecByPath(specs, testEmbeddingModelPath)
	if !ok {
		t.Fatalf("BuildModelSpecs() produced no spec for %q; got %#v", testEmbeddingModelPath, specs)
	}
	for _, want := range embeddingModelWeightFiles {
		if !slices.Contains(spec.RequiredFiles, want) {
			t.Fatalf("spec RequiredFiles = %#v, missing %q", spec.RequiredFiles, want)
		}
	}
}

// TestAliasedEmbeddingModelStillSkippedForRemoteBackend keeps the feature gates keyed the same
// way as the collector. A gate that compares the literal config value against a canonicalized
// path stops matching, and a deployment that embeds remotely downloads a model it never loads.
func TestAliasedEmbeddingModelStillSkippedForRemoteBackend(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: config.ToLegacyRegistry(),
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				MmBertModelPath: "models/mom-embedding-ultra",
				EmbeddingConfig: config.HNSWConfig{
					Backend:   config.EmbeddingBackendOpenAICompatible,
					ModelType: config.EmbeddingModelTypeRemote,
				},
				Endpoint: config.EmbeddingEndpointConfig{
					BaseURL: "http://embedding-service:8000/v1",
					Model:   "BAAI/bge-m3",
				},
			},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("BuildModelSpecs() returned %d specs for a remote embedding backend, want 0: %#v",
			len(specs), specs)
	}
}

// TestBuildModelSpecsKeepsClassifierModelIDLiteral pins the other half of the contract. The
// built-in classifiers pass model_id straight to the bindings without resolving it, so
// canonicalizing it here would download the snapshot to a directory they never open.
func TestBuildModelSpecsKeepsClassifierModelIDLiteral(t *testing.T) {
	const aliasModelID = "models/category_classifier_modernbert-base_model"

	cfg := &config.RouterConfig{
		MoMRegistry: config.ToLegacyRegistry(),
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             aliasModelID,
					CategoryMappingPath: aliasModelID + "/category_mapping.json",
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name: "domain-route",
				Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
					{Type: config.SignalTypeDomain, Name: "billing"},
				}},
			}},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 1: %#v", len(specs), specs)
	}
	if specs[0].LocalPath != aliasModelID {
		t.Fatalf("BuildModelSpecs() downloads %q, but the classifier loads %q verbatim",
			specs[0].LocalPath, aliasModelID)
	}
	if !slices.Contains(specs[0].RequiredFiles, "category_mapping.json") {
		t.Fatalf("spec RequiredFiles = %#v, missing %q", specs[0].RequiredFiles, "category_mapping.json")
	}
}
