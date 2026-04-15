package modeldownload

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var expectedAMDModelSpecs = []string{
	"models/mmbert-embed-32k-2d-matryoshka",
	"models/mmbert32k-intent-classifier-merged",
	"models/mmbert32k-factcheck-classifier-merged",
	"models/mmbert32k-feedback-detector-merged",
}

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
			expected: []string{"models/lora_intent_classifier_bert-base-uncased_model"},
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

func TestBuildModelSpecsSkipsDisabledHallucinationFeatureModels(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-factcheck-classifier-merged": "llm-semantic-router/mmbert32k-factcheck-classifier-merged",
			"models/mom-halugate-detector":                 "llm-semantic-router/mom-halugate-detector",
			"models/mom-halugate-explainer":                "llm-semantic-router/mom-halugate-explainer",
		},
		InlineModels: config.InlineModels{
			HallucinationMitigation: config.HallucinationMitigationConfig{
				Enabled: false,
				FactCheckModel: config.FactCheckModelConfig{
					ModelID: "models/mmbert32k-factcheck-classifier-merged",
				},
				HallucinationModel: config.HallucinationModelConfig{
					ModelID: "models/mom-halugate-detector",
				},
				NLIModel: config.NLIModelConfig{
					ModelID: "models/mom-halugate-explainer",
				},
			},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 0", len(specs))
	}
}

func TestBuildModelSpecsIncludesFactCheckClassifierWhenSignalConfigured(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-factcheck-classifier-merged": "llm-semantic-router/mmbert32k-factcheck-classifier-merged",
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				FactCheckRules: []config.FactCheckRule{
					{Name: "needs_fact_check"},
				},
			},
		},
		InlineModels: config.InlineModels{
			HallucinationMitigation: config.HallucinationMitigationConfig{
				FactCheckModel: config.FactCheckModelConfig{
					ModelID: "models/mmbert32k-factcheck-classifier-merged",
				},
			},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 1 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 1", len(specs))
	}
	if specs[0].LocalPath != "models/mmbert32k-factcheck-classifier-merged" {
		t.Fatalf("LocalPath = %q, want fact-check classifier", specs[0].LocalPath)
	}
}

func TestBuildModelSpecsSkipsUnusedCoreClassifierModels(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-intent-classifier-merged":  "llm-semantic-router/mmbert32k-intent-classifier-merged",
			"models/mmbert32k-pii-detector-merged":       "llm-semantic-router/mmbert32k-pii-detector-merged",
			"models/mmbert32k-jailbreak-detector-merged": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
		},
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
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name:  "default-route",
				Rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{}},
			}},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 0", len(specs))
	}
}

func TestBuildModelSpecsIncludesUsedCoreClassifierModels(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-intent-classifier-merged":  "llm-semantic-router/mmbert32k-intent-classifier-merged",
			"models/mmbert32k-pii-detector-merged":       "llm-semantic-router/mmbert32k-pii-detector-merged",
			"models/mmbert32k-jailbreak-detector-merged": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
		},
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
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{{
				Name: "guarded-route",
				Rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
					{Type: config.SignalTypeDomain, Name: "billing"},
					{Type: config.SignalTypePII, Name: "contains_pii"},
					{Type: config.SignalTypeJailbreak, Name: "detector"},
				}},
			}},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 3 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 3", len(specs))
	}
	assertContainsAllModelSpecs(t, specs,
		"models/mmbert32k-intent-classifier-merged",
		"models/mmbert32k-pii-detector-merged",
		"models/mmbert32k-jailbreak-detector-merged",
	)
}

func TestBuildModelSpecsIncludesCoreClassifierUsedViaProjection(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-jailbreak-detector-merged": "llm-semantic-router/mmbert32k-jailbreak-detector-merged",
		},
		InlineModels: config.InlineModels{
			PromptGuard: config.PromptGuardConfig{
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Projections: config.Projections{
				Scores: []config.ProjectionScore{{
					Name:   "risk_score",
					Method: "weighted_sum",
					Inputs: []config.ProjectionScoreInput{{Type: config.SignalTypeJailbreak, Name: "detector", Weight: 1.0}},
				}},
				Mappings: []config.ProjectionMapping{{
					Name:   "risk_map",
					Source: "risk_score",
					Method: "threshold",
					Outputs: []config.ProjectionMappingOutput{{
						Name: "high_risk",
					}},
				}},
			},
			Decisions: []config.Decision{{
				Name:  "guarded-route",
				Rules: config.RuleNode{Type: config.SignalTypeProjection, Name: "high_risk"},
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
	if specs[0].LocalPath != "models/mmbert32k-jailbreak-detector-merged" {
		t.Fatalf("LocalPath = %q, want jailbreak classifier", specs[0].LocalPath)
	}
}

func TestBuildModelSpecsIncludesRouterOwnedDefaultsForScratchCanonicalConfig(t *testing.T) {
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
          weight: 100
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
      modelRefs:
        - model: openai/gpt-oss-120b
          use_reasoning: false
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	assertContainsAllModelSpecs(t, specs,
		"models/mmbert-embed-32k-2d-matryoshka",
	)
}

func TestBuildModelSpecsIncludesRouterOwnedDefaultsForSparseAMDGlobalOverride(t *testing.T) {
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
          weight: 100
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
      modelRefs:
        - model: openai/gpt-oss-120b
          use_reasoning: false
global:
  model_catalog:
    embeddings:
      semantic:
        use_cpu: false
    modules:
      prompt_guard:
        use_cpu: false
      classifier:
        domain:
          use_cpu: false
        pii:
          use_cpu: false
      hallucination_mitigation:
        fact_check:
          use_cpu: false
        detector:
          use_cpu: false
        explainer:
          use_cpu: false
      feedback_detector:
        use_cpu: false
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	assertContainsAllModelSpecs(t, specs,
		"models/mmbert-embed-32k-2d-matryoshka",
	)
}

func TestBuildModelSpecsSkipsUnusedFeedbackDetectorDefaults(t *testing.T) {
	cfg := &config.RouterConfig{
		MoMRegistry: map[string]string{
			"models/mmbert32k-feedback-detector-merged": "llm-semantic-router/mmbert32k-feedback-detector-merged",
		},
		InlineModels: config.InlineModels{
			FeedbackDetector: config.FeedbackDetectorConfig{
				Enabled: true,
				ModelID: "models/mmbert32k-feedback-detector-merged",
			},
		},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}
	if len(specs) != 0 {
		t.Fatalf("BuildModelSpecs() returned %d specs, want 0", len(specs))
	}
}

func TestBuildModelSpecsAcceptsReferenceConfig(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve reference config path")
	}

	configPath := filepath.Clean(filepath.Join(filepath.Dir(file), "../../../../config/config.yaml"))
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read %s: %v", configPath, err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	assertContainsAllModelSpecs(t, specs,
		"models/mom-embedding-pro",
		"models/mom-embedding-flash",
		"models/mmbert-embed-32k-2d-matryoshka",
		"models/mom-embedding-light",
		"models/mmbert32k-modality-router-merged",
	)
}

func TestBuildModelSpecsIncludesAllAMDDeployModels(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve amd config path")
	}

	configPath := filepath.Clean(filepath.Join(filepath.Dir(file), "../../../../deploy/recipes/balance.yaml"))
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read %s: %v", configPath, err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatalf("BuildModelSpecs() error = %v", err)
	}

	assertContainsAllModelSpecs(t, specs,
		expectedAMDModelSpecs...,
	)
	if len(specs) != len(expectedAMDModelSpecs) {
		t.Fatalf("BuildModelSpecs() returned %d specs, want %d", len(specs), len(expectedAMDModelSpecs))
	}
}

func TestBuildModelSpecsSkipsRouterOwnedDefaultsForAgentSmokeConfigs(t *testing.T) {
	for _, relParts := range [][]string{
		{"..", "..", "..", "..", "e2e", "config", "config.agent-smoke.cpu.yaml"},
		{"..", "..", "..", "..", "e2e", "config", "config.agent-smoke.amd.yaml"},
	} {
		rel := filepath.Join(relParts...)
		t.Run(rel, func(t *testing.T) {
			_, file, _, ok := runtime.Caller(0)
			if !ok {
				t.Fatal("failed to resolve smoke config path")
			}

			configPath := filepath.Clean(filepath.Join(filepath.Dir(file), rel))
			data, err := os.ReadFile(configPath)
			if err != nil {
				t.Fatalf("read %s: %v", configPath, err)
			}

			cfg, err := config.ParseYAMLBytes(data)
			if err != nil {
				t.Fatalf("ParseYAMLBytes() error = %v", err)
			}

			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatalf("BuildModelSpecs() error = %v", err)
			}
			if len(specs) != 0 {
				t.Fatalf("BuildModelSpecs() returned %d specs, want 0: %#v", len(specs), specs)
			}
		})
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
