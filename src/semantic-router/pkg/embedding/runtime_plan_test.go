package embedding

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolveRuntimePlanLocalModelMatrix(t *testing.T) {
	tests := []struct {
		name      string
		backend   string
		modelType string
		wantModel string
		wantError string
	}{
		{name: "candle qwen3", backend: config.EmbeddingBackendCandle, modelType: "qwen3", wantModel: "qwen3"},
		{name: "candle gemma", backend: config.EmbeddingBackendCandle, modelType: "gemma", wantModel: "gemma"},
		{name: "candle mmbert", backend: config.EmbeddingBackendCandle, modelType: "mmbert", wantModel: "mmbert"},
		{name: "candle modernbert canonical", backend: config.EmbeddingBackendCandle, modelType: "modernbert", wantModel: "mmbert"},
		{name: "candle multimodal", backend: config.EmbeddingBackendCandle, modelType: "multimodal", wantModel: "multimodal"},
		{name: "openvino qwen3", backend: config.EmbeddingBackendOpenVINO, modelType: "qwen3", wantModel: "qwen3"},
		{name: "openvino gemma", backend: config.EmbeddingBackendOpenVINO, modelType: "gemma", wantModel: "gemma"},
		{name: "openvino mmbert", backend: config.EmbeddingBackendOpenVINO, modelType: "mmbert", wantModel: "mmbert"},
		{name: "openvino multimodal rejected", backend: config.EmbeddingBackendOpenVINO, modelType: "multimodal", wantError: "does not support"},
		{name: "unknown local model rejected", backend: config.EmbeddingBackendCandle, modelType: "mystery", wantError: "does not support"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			models := localRuntimePlanModels(tt.backend, tt.modelType)
			plan, err := ResolveRuntimePlan(models, "", "")
			if tt.wantError != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("ResolveRuntimePlan error = %v, want %q", err, tt.wantError)
				}
				return
			}
			if err != nil {
				t.Fatalf("ResolveRuntimePlan: %v", err)
			}
			if plan.Backend != tt.backend || plan.ModelType != tt.wantModel {
				t.Fatalf("plan = %+v, want backend %q model %q", plan, tt.backend, tt.wantModel)
			}
		})
	}
}

func TestResolveRuntimePlanRemoteToLocalRequiresUnambiguousModel(t *testing.T) {
	remote := config.EmbeddingModels{EmbeddingConfig: config.HNSWConfig{
		Backend:   config.EmbeddingBackendOpenAICompatible,
		ModelType: config.EmbeddingModelTypeRemote,
	}}
	if _, err := ResolveRuntimePlan(remote, config.EmbeddingBackendCandle, ""); err == nil {
		t.Fatal("remote-to-local plan without a local model succeeded")
	}

	remote.Qwen3ModelPath = "models/qwen3"
	remote.MmBertModelPath = "models/mmbert"
	if _, err := ResolveRuntimePlan(remote, config.EmbeddingBackendCandle, ""); err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("ambiguous remote-to-local error = %v", err)
	}

	plan, err := ResolveRuntimePlan(remote, config.EmbeddingBackendCandle, "mmbert")
	if err != nil {
		t.Fatalf("explicit remote-to-local plan: %v", err)
	}
	if !plan.LocalOverride || plan.Backend != config.EmbeddingBackendCandle || plan.ModelType != "mmbert" || plan.ModelPath != remote.MmBertModelPath {
		t.Fatalf("explicit plan = %+v", plan)
	}
}

func TestResolveRuntimePlanRejectsUnsupportedBackend(t *testing.T) {
	_, err := ResolveRuntimePlan(localRuntimePlanModels("unsupported", "qwen3"), "", "")
	if err == nil || !strings.Contains(err.Error(), "unsupported embedding backend") {
		t.Fatalf("unsupported backend error = %v", err)
	}
}

func TestModelTypeOverrideFromEnvPreservesLegacyContract(t *testing.T) {
	t.Setenv("EMBEDDING_MODEL_OVERRIDE", " multimodal ")
	t.Setenv(embeddingModelTypeOverrideEnv, "")
	if got := ModelTypeOverrideFromEnv(); got != "multimodal" {
		t.Fatalf("legacy model override = %q, want multimodal", got)
	}

	t.Setenv(embeddingModelTypeOverrideEnv, " MMBERT ")
	if got := ModelTypeOverrideFromEnv(); got != "mmbert" {
		t.Fatalf("explicit model-type override = %q, want mmbert", got)
	}
}

func localRuntimePlanModels(backend string, modelType string) config.EmbeddingModels {
	return config.EmbeddingModels{
		Qwen3ModelPath:      "models/qwen3",
		GemmaModelPath:      "models/gemma",
		MmBertModelPath:     "models/mmbert",
		MultiModalModelPath: "models/multimodal",
		EmbeddingConfig: config.HNSWConfig{
			Backend:   backend,
			ModelType: modelType,
		},
	}
}
