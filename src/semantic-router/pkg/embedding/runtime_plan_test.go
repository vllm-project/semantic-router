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

func TestResolveRuntimePlanLocalBackendsRequireSelectedModelPath(t *testing.T) {
	for _, backend := range []string{config.EmbeddingBackendCandle, config.EmbeddingBackendOpenVINO} {
		for _, modelType := range []string{config.EmbeddingModelTypeQwen3, "gemma", "mmbert"} {
			t.Run(backend+"/"+modelType, func(t *testing.T) {
				models := localRuntimePlanModels(backend, modelType)
				switch modelType {
				case config.EmbeddingModelTypeQwen3:
					models.Qwen3ModelPath = ""
				case "gemma":
					models.GemmaModelPath = ""
				case "mmbert":
					models.MmBertModelPath = ""
				}

				_, err := ResolveRuntimePlan(models, "", "")
				if err == nil || !strings.Contains(err.Error(), "requires a configured "+modelType+" model path") {
					t.Fatalf("ResolveRuntimePlan error = %v, want missing selected model path", err)
				}
			})
		}
	}
}

func TestResolveRuntimePlanLocalToRemoteUsesRemoteModelContract(t *testing.T) {
	models := localRuntimePlanModels(config.EmbeddingBackendCandle, config.EmbeddingModelTypeQwen3)
	models.Endpoint = config.EmbeddingEndpointConfig{
		BaseURL: "https://embedding.example/v1",
		Model:   "remote-embedding",
	}

	plan, err := ResolveRuntimePlan(models, config.EmbeddingBackendOpenAICompatible, "")
	if err != nil {
		t.Fatalf("local-to-remote plan: %v", err)
	}
	if plan.Backend != config.EmbeddingBackendOpenAICompatible || plan.ModelType != config.EmbeddingModelTypeRemote || plan.LocalOverride {
		t.Fatalf("local-to-remote plan = %+v", plan)
	}
}

func TestResolveRuntimePlanLocalToRemoteRequiresEndpointCapability(t *testing.T) {
	models := localRuntimePlanModels(config.EmbeddingBackendCandle, config.EmbeddingModelTypeQwen3)

	_, err := ResolveRuntimePlan(models, config.EmbeddingBackendOpenAICompatible, "")
	if err == nil || !strings.Contains(err.Error(), "usable endpoint") {
		t.Fatalf("local-to-remote endpoint error = %v", err)
	}
}

func TestResolveRuntimePlanRejectsUnsupportedBackend(t *testing.T) {
	_, err := ResolveRuntimePlan(localRuntimePlanModels("unsupported", "qwen3"), "", "")
	if err == nil || !strings.Contains(err.Error(), "unsupported embedding backend") {
		t.Fatalf("unsupported backend error = %v", err)
	}
}

func TestResolveRuntimePlansPreservesDistinctConsumerModelsAndDeduplicates(t *testing.T) {
	models := localRuntimePlanModels(config.EmbeddingBackendCandle, config.EmbeddingModelTypeQwen3)
	plans, err := ResolveRuntimePlans(models, "", "", "qwen3", "mmbert", "modernbert", "qwen3")
	if err != nil {
		t.Fatalf("ResolveRuntimePlans: %v", err)
	}
	want := []RuntimePlan{
		{Backend: config.EmbeddingBackendCandle, ModelType: "qwen3", ModelPath: models.Qwen3ModelPath},
		{Backend: config.EmbeddingBackendCandle, ModelType: "mmbert", ModelPath: models.MmBertModelPath},
	}
	if len(plans) != len(want) {
		t.Fatalf("plans = %+v, want %+v", plans, want)
	}
	for index := range want {
		if plans[index] != want[index] {
			t.Fatalf("plans[%d] = %+v, want %+v", index, plans[index], want[index])
		}
	}
}

func TestResolveRuntimePlansModelOverrideCollapsesConsumerModels(t *testing.T) {
	models := localRuntimePlanModels(config.EmbeddingBackendCandle, config.EmbeddingModelTypeQwen3)
	plans, err := ResolveRuntimePlans(models, "", "mmbert", "qwen3", "gemma")
	if err != nil {
		t.Fatalf("ResolveRuntimePlans: %v", err)
	}
	if len(plans) != 1 || plans[0].ModelType != "mmbert" || plans[0].ModelPath != models.MmBertModelPath {
		t.Fatalf("override plans = %+v, want one mmbert plan", plans)
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
