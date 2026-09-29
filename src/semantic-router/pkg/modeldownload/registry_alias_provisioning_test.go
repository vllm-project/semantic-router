package modeldownload

import (
	"fmt"
	"reflect"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildModelSpecsCanonicalEmbeddingAliases(t *testing.T) {
	requireCandleEmbeddingRuntime(t)
	for _, tc := range []struct {
		family string
		field  string
		alias  string
	}{
		{"qwen3", "qwen3_model_path", "qwen3"},
		{"gemma", "gemma_model_path", "gemma"},
		{"mmbert", "mmbert_model_path", "embedding-ultra"},
		{"mmbert", "mmbert_model_path", "Vela-1.0-Encoder-307M-Embedding"},
	} {
		t.Run(tc.alias, func(t *testing.T) {
			canonical := config.ResolveModelPath(tc.alias)
			if canonical == tc.alias {
				t.Fatalf("fixture alias is not registered: %s", tc.alias)
			}
			var baseline ModelSpec
			for _, declared := range slices.Compact([]string{canonical, "models/" + tc.alias, tc.alias}) {
				t.Run(declared, func(t *testing.T) {
					cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(embeddingAliasConfigYAML, tc.field, declared, tc.family)))
					if err != nil {
						t.Fatal(err)
					}
					specs, err := BuildModelSpecs(cfg)
					if err != nil {
						t.Fatal(err)
					}
					spec, found := findSpecByPath(specs, canonical)
					if !found {
						t.Fatalf("public embedding API config omitted required artifact %q: %+v", canonical, specs)
					}
					if declared == canonical {
						baseline = spec
					} else if !reflect.DeepEqual(spec, baseline) {
						t.Errorf("alias download contract = %+v, canonical = %+v", spec, baseline)
					}
					files := []string{"model.safetensors", "tokenizer.json"}
					if tc.family == "gemma" {
						files = append(files, "2_Dense/model.safetensors", "3_Dense/model.safetensors")
					}
					for _, file := range files {
						if !slices.Contains(spec.RequiredFiles, file) {
							t.Errorf("required Candle file %q absent from %v", file, spec.RequiredFiles)
						}
					}
				})
			}
		})
	}
}

const embeddingAliasConfigYAML = `version: v0.3
providers:
  defaults:
    model: demo
  models:
    - name: demo
      backend_refs:
        - name: primary
          endpoint: localhost:8000
          protocol: http
          weight: 1
global:
  services:
    api:
      embeddings:
        enabled: true
  model_catalog:
    embeddings:
      semantic:
        %s: %s
        use_cpu: true
        embedding_config:
          model_type: %s
`

func TestExtractModelPathsCanonicalAliasScope(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Qwen3ModelPath = "qwen3"
	cfg.GemmaModelPath = "models/qwen3"
	cfg.MmBertModelPath = "models/mom-embedding-pro"
	cfg.BertModelPath = "example/unregistered-model"
	cfg.MultiModalModelPath = "/opt/local-model"
	cfg.CategoryModel.ModelID = "models/local-weights.safetensors"
	if got := ExtractModelPaths(cfg); !reflect.DeepEqual(got, []string{"models/mom-embedding-pro"}) {
		t.Fatalf("alias deduplication or nonlocal/file scope changed: %v", got)
	}
}

func TestAliasedModelRequirementsShareCanonicalKey(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.GemmaModelPath = "gemma"
	cfg.MmBertModelPath = "models/gemma"
	required := candleEmbeddingModelRequiredFiles(cfg)
	if _, exists := required["gemma"]; exists {
		t.Fatal("Candle requirements retained a bare alias key")
	}
	if _, exists := required["models/gemma"]; exists {
		t.Fatal("Candle requirements retained a prefixed alias key")
	}
	want := []string{"model.safetensors", "tokenizer.json", "2_Dense/model.safetensors", "3_Dense/model.safetensors"}
	if got := required["models/mom-embedding-flash"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("merged Candle requirements = %v, want %v", got, want)
	}
	cfg.CategoryMappingPath = "models/domain-classifier/category_mapping.json"
	if got := ExtractRequiredFilesByModel(cfg)["models/mom-domain-classifier"]; !reflect.DeepEqual(got, []string{"category_mapping.json"}) {
		t.Fatalf("mapping requirements lost canonical identity: %v", got)
	}
}

func TestBuildModelSpecsAliasedFeatureGate(t *testing.T) {
	for _, active := range []bool{false, true} {
		t.Run(fmt.Sprint(active), func(t *testing.T) {
			cfg := &config.RouterConfig{MoMRegistry: config.ToLegacyRegistry()}
			cfg.CategoryModel.ModelID = "domain-classifier"
			if active {
				cfg.ClassifierRules = []config.ClassifierSignalRule{{
					Name: "topic", Type: "local", ModelPath: "models/mom-domain-classifier", Labels: []string{"general"},
				}}
			}
			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatal(err)
			}
			_, found := findSpecByPath(specs, "models/mom-domain-classifier")
			if found != active {
				t.Fatalf("alias feature gate found=%v, active=%v: %+v", found, active, specs)
			}
		})
	}
}

func TestBuildModelSpecsRemoteProviderAliasIsNotLocalArtifact(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
providers:
  defaults:
    model: remote
  models:
    - name: remote
      provider_model_id: qwen3
      backend_refs:
        - name: primary
          endpoint: localhost:8000
          protocol: http
          weight: 1
global:
  services:
    api:
      embeddings:
        enabled: false
`))
	if err != nil {
		t.Fatal(err)
	}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, found := findSpecByPath(specs, "models/mom-embedding-pro"); found {
		t.Fatalf("remote provider alias caused local model downloads: %+v", specs)
	}
}
