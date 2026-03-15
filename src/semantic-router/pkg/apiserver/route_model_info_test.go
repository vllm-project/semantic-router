//go:build !windows && cgo

package apiserver

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func requireModelInfo(t *testing.T, models []ModelInfo, name string) ModelInfo {
	t.Helper()

	for _, model := range models {
		if model.Name == name {
			return model
		}
	}

	t.Fatalf("expected %s entry, got %+v", name, models)
	return ModelInfo{}
}

func requireReadyModel(t *testing.T, models map[string]ModelInfo, name, path string) {
	t.Helper()

	model, ok := models[name]
	if !ok {
		t.Fatalf("expected %s in models response, got %+v", name, models)
	}
	if model.ModelPath != path {
		t.Fatalf("expected %s path %q, got %q", name, path, model.ModelPath)
	}
	if !model.Loaded {
		t.Fatalf("expected %s to be marked loaded", name)
	}
	if model.State != "ready" {
		t.Fatalf("expected %s state ready, got %q", name, model.State)
	}
	if model.Registry == nil {
		t.Fatalf("expected registry metadata for %s", name)
	}
}

func buildAuxiliaryModelsConfig() *config.RouterConfig {
	return &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					Threshold:           0.42,
					UseMmBERT32K:        true,
					CategoryMappingPath: "models/mmbert32k-intent-classifier-merged/category_mapping.json",
				},
				PIIModel: config.PIIModel{
					ModelID:        "models/mmbert32k-pii-detector-merged",
					Threshold:      0.73,
					UseMmBERT32K:   true,
					PIIMappingPath: "models/mmbert32k-pii-detector-merged/label_mapping.json",
				},
			},
			PromptGuard: config.PromptGuardConfig{
				Enabled:              true,
				ModelID:              "models/mmbert32k-jailbreak-detector-merged",
				UseMmBERT32K:         true,
				JailbreakMappingPath: "models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
			},
			HallucinationMitigation: config.HallucinationMitigationConfig{
				Enabled: true,
				FactCheckModel: config.FactCheckModelConfig{
					ModelID:      "models/mmbert32k-factcheck-classifier-merged",
					Threshold:    0.61,
					UseCPU:       true,
					UseMmBERT32K: true,
				},
				HallucinationModel: config.HallucinationModelConfig{
					ModelID:                "models/mom-halugate-detector",
					Threshold:              0.80,
					UseCPU:                 true,
					MinSpanLength:          2,
					MinSpanConfidence:      0.60,
					ContextWindowSize:      50,
					EnableNLIFiltering:     true,
					NLIEntailmentThreshold: 0.75,
				},
				NLIModel: config.NLIModelConfig{
					ModelID:   "models/mom-halugate-explainer",
					Threshold: 0.90,
					UseCPU:    true,
				},
			},
			FeedbackDetector: config.FeedbackDetectorConfig{
				Enabled:      true,
				ModelID:      "models/mmbert32k-feedback-detector-merged",
				Threshold:    0.70,
				UseCPU:       true,
				UseMmBERT32K: true,
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				Categories: []config.Category{
					{CategoryMetadata: config.CategoryMetadata{Name: "billing"}},
				},
			},
		},
	}
}

func TestBuildModelsInfoResponseIncludesRuntimeSummaryAndRegistryMetadata(t *testing.T) {
	t.Parallel()

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "models/mmbert32k-intent-classifier-merged",
					Threshold:           0.42,
					CategoryMappingPath: "live-mapping.json",
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				Categories: []config.Category{
					{CategoryMetadata: config.CategoryMetadata{Name: "billing"}},
				},
			},
		},
	}

	configPath := filepath.Join(t.TempDir(), "router-config.yaml")
	if err := startupstatus.NewWriter(configPath).Write(startupstatus.State{
		Phase:            "downloading_models",
		Ready:            false,
		Message:          "Downloading model models/mmbert32k-intent-classifier-merged",
		DownloadingModel: "models/mmbert32k-intent-classifier-merged",
		PendingModels:    []string{"models/mmbert32k-intent-classifier-merged"},
		ReadyModels:      0,
		TotalModels:      1,
	}); err != nil {
		t.Fatalf("write runtime state: %v", err)
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            cfg,
		runtimeConfig: newLiveRuntimeConfig(
			cfg,
			func() *config.RouterConfig { return cfg },
			nil,
		),
		configPath: configPath,
	}

	resp := apiServer.buildModelsInfoResponse()
	if resp.Summary.Phase != "downloading_models" {
		t.Fatalf("expected downloading_models phase, got %q", resp.Summary.Phase)
	}
	if resp.Summary.TotalModels != 1 {
		t.Fatalf("expected summary total models 1, got %d", resp.Summary.TotalModels)
	}
	if resp.Summary.LoadedModels != 0 {
		t.Fatalf("expected summary loaded models 0, got %d", resp.Summary.LoadedModels)
	}

	categoryModel := requireModelInfo(t, resp.Models, "category_classifier")
	if categoryModel.Loaded {
		t.Fatalf("expected category model to be unloaded during startup")
	}
	if categoryModel.State != "downloading" {
		t.Fatalf("expected category model state downloading, got %q", categoryModel.State)
	}
	if categoryModel.Registry == nil {
		t.Fatalf("expected registry metadata for category model")
	}
	if categoryModel.Registry.RepoID == "" {
		t.Fatalf("expected registry repo id, got %+v", categoryModel.Registry)
	}
	if categoryModel.Registry.LocalPath != "models/mmbert32k-intent-classifier-merged" {
		t.Fatalf("expected canonical local path, got %+v", categoryModel.Registry)
	}
}

func TestBuildModelsInfoResponseMarksLoadedClassifierModelsReady(t *testing.T) {
	t.Parallel()

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			Classifier: config.Classifier{
				CategoryModel: config.CategoryModel{
					ModelID:             "live-category-model",
					Threshold:           0.42,
					CategoryMappingPath: "live-mapping.json",
				},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				Categories: []config.Category{
					{CategoryMetadata: config.CategoryMetadata{Name: "billing"}},
				},
			},
		},
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: &fakeResolvedClassificationService{},
		config:            cfg,
		runtimeConfig: newLiveRuntimeConfig(
			cfg,
			func() *config.RouterConfig { return cfg },
			nil,
		),
	}

	resp := apiServer.buildModelsInfoResponse()
	if !resp.Summary.Ready {
		t.Fatalf("expected summary ready when classifier is available, got %+v", resp.Summary)
	}

	categoryModel := requireModelInfo(t, resp.Models, "category_classifier")
	if !categoryModel.Loaded {
		t.Fatalf("expected category model to be loaded")
	}
	if categoryModel.State != "ready" {
		t.Fatalf("expected category model state ready, got %q", categoryModel.State)
	}
}

func TestBuildModelsInfoResponseIncludesConfiguredAuxiliaryModels(t *testing.T) {
	t.Parallel()

	cfg := buildAuxiliaryModelsConfig()

	apiServer := &ClassificationAPIServer{
		classificationSvc: &fakeResolvedClassificationService{},
		config:            cfg,
		runtimeConfig: newLiveRuntimeConfig(
			cfg,
			func() *config.RouterConfig { return cfg },
			nil,
		),
	}

	resp := apiServer.buildModelsInfoResponse()
	expected := map[string]string{
		"category_classifier":     "models/mmbert32k-intent-classifier-merged",
		"pii_classifier":          "models/mmbert32k-pii-detector-merged",
		"jailbreak_classifier":    "models/mmbert32k-jailbreak-detector-merged",
		"fact_check_classifier":   "models/mmbert32k-factcheck-classifier-merged",
		"hallucination_detector":  "models/mom-halugate-detector",
		"hallucination_explainer": "models/mom-halugate-explainer",
		"feedback_detector":       "models/mmbert32k-feedback-detector-merged",
	}

	modelsByName := map[string]ModelInfo{}
	for _, model := range resp.Models {
		modelsByName[model.Name] = model
	}

	for name, path := range expected {
		requireReadyModel(t, modelsByName, name, path)
	}
}

func TestNormalizeEmbeddingModelPathExtractsRegistryPathFromRuntimeString(t *testing.T) {
	t.Parallel()

	runtimePath := "MmBertEmbeddingModel(path=models/mom-embedding-ultra, hidden_size=768, layers=22)"
	if got := normalizeEmbeddingModelPath(runtimePath, "mmbert"); got != "models/mom-embedding-ultra" {
		t.Fatalf("expected normalized embedding path, got %q", got)
	}
}

func TestNormalizeEmbeddingModelPathFallsBackToRegistryAlias(t *testing.T) {
	t.Parallel()

	if got := normalizeEmbeddingModelPath("", "gemma"); got != "models/mom-embedding-flash" {
		t.Fatalf("expected alias to resolve to registry local path, got %q", got)
	}
}
