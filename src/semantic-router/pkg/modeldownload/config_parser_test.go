package modeldownload

import (
	"reflect"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestExtractModelPaths(t *testing.T) {
	tests := []struct {
		name     string
		config   *config.RouterConfig
		expected []string
	}{
		{
			name: "Extract Qwen3ModelPath",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{
						Qwen3ModelPath: "models/mom-embedding-pro",
					},
				},
			},
			expected: []string{"models/mom-embedding-pro"},
		},
		{
			name: "Extract GemmaModelPath",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{
						GemmaModelPath: "models/mom-embedding-flash",
					},
				},
			},
			expected: []string{"models/mom-embedding-flash"},
		},
		{
			name: "Extract both embedding models",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{
						Qwen3ModelPath: "models/mom-embedding-pro",
						GemmaModelPath: "models/mom-embedding-flash",
					},
				},
			},
			expected: []string{"models/mom-embedding-pro", "models/mom-embedding-flash"},
		},
		{
			name: "Extract ModelID from classifier",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					Classifier: config.Classifier{
						CategoryModel: config.CategoryModel{
							ModelID: "models/lora_intent_classifier_bert-base-uncased_model",
						},
					},
				},
			},
			expected: []string{"models/mom-domain-classifier"},
		},
		{
			name: "Extract multiple model paths",
			config: &config.RouterConfig{
				InlineModels: config.InlineModels{
					EmbeddingModels: config.EmbeddingModels{
						Qwen3ModelPath: "models/mom-embedding",
					},
					Classifier: config.Classifier{
						CategoryModel: config.CategoryModel{
							ModelID: "models/mom-domain-classifier",
						},
					},
				},
			},
			expected: []string{"models/mom-embedding", "models/mom-domain-classifier"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			paths := ExtractModelPaths(tt.config)

			if len(paths) != len(tt.expected) {
				t.Errorf("Expected %d paths, got %d: %v", len(tt.expected), len(paths), paths)
				return
			}

			// Check if all expected paths are present (order doesn't matter)
			pathMap := make(map[string]bool)
			for _, p := range paths {
				pathMap[p] = true
			}

			for _, expected := range tt.expected {
				if !pathMap[expected] {
					t.Errorf("Expected path %s not found in result: %v", expected, paths)
				}
			}
		})
	}
}

func TestIsModelDirectory(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"models/bert-base-uncased", true},
		{"models/gmtrouter.pt", false},
		{"models/lora_model/adapter_config.json", false},
		{"models/mapping.json", false},
		{"config/tools_db.json", false},
		{"models/mom-embedding-pro", true},
		{"models/mom-embedding-flash", true},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			result := isModelDirectory(tt.path)
			if result != tt.expected {
				t.Errorf("isModelDirectory(%s) = %v, expected %v", tt.path, result, tt.expected)
			}
		})
	}
}

func TestExtractModelPathsSkipsRootLevelModelFiles(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http-8888
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    default_model: openai/gpt-oss-120b
  models:
    - name: openai/gpt-oss-120b
      provider_model_id: openai/gpt-oss-120b
      backend_refs:
        - name: primary
          endpoint: localhost:8000
          protocol: http
routing:
  modelCards:
    - name: openai/gpt-oss-120b
      modality: text
  decisions:
    - name: default-route
      priority: 100
      rules:
        operator: AND
        conditions: []
      algorithm:
        type: gmtrouter
        gmtrouter:
          model_path: models/gmtrouter.pt
      modelRefs:
        - model: openai/gpt-oss-120b
          use_reasoning: false
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	if got := ExtractModelPaths(cfg); slices.Contains(got, "models/gmtrouter.pt") {
		t.Fatalf("ExtractModelPaths() unexpectedly included root-level model file: %v", got)
	}
}

func TestExtractRequiredFilesByModel(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
				},
				PIIModel: config.PIIModel{
					ModelID:        "models/mmbert32k-pii-detector-merged",
					PIIMappingPath: "models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
				},
			},
			PromptGuard: config.PromptGuardConfig{
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
			},
		},
	}

	got := ExtractRequiredFilesByModel(cfg)
	want := map[string][]string{
		"models/mmbert32k-intent-classifier-merged":  {"category_mapping.json"},
		"models/mmbert32k-pii-detector-merged":       {"pii_type_mapping.json"},
		"models/mmbert32k-jailbreak-detector-merged": {"jailbreak_type_mapping.json"},
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ExtractRequiredFilesByModel() = %#v, want %#v", got, want)
	}
}

func TestBuildModelSpecsIncludesConfigDerivedRequiredFiles(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-intent-classifier-merged": "llm-semantic-router/mmbert32k-intent-classifier-merged",
		},
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "domain-route",
				Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"},
			}},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 1", len(specs))
	}

	wantRequiredFiles := []string{"config.json", "category_mapping.json"}
	if !reflect.DeepEqual(specs[0].RequiredFiles, wantRequiredFiles) {
		t.Fatalf("RequiredFiles = %#v, want %#v", specs[0].RequiredFiles, wantRequiredFiles)
	}
}

func assertContainsAllModelSpecs(t *testing.T, specs []ModelSpec, wantPaths ...string) {
	t.Helper()

	gotPaths := make([]string, 0, len(specs))
	for _, spec := range specs {
		gotPaths = append(gotPaths, spec.LocalPath)
	}

	for _, want := range wantPaths {
		if !slices.Contains(gotPaths, want) {
			t.Fatalf("BuildModelSpecs() missing %q; got %v", want, gotPaths)
		}
	}
}
