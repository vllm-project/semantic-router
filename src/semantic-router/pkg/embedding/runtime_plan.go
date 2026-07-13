package embedding

import (
	"fmt"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const embeddingModelTypeOverrideEnv = "EMBEDDING_MODEL_TYPE_OVERRIDE"

// ModelTypeOverrideFromEnv reads the explicit model-type selection used when a
// remote configuration is switched to a local backend. The unambiguous name
// takes precedence while the deployed EMBEDDING_MODEL_OVERRIDE contract stays
// compatible with existing Helm and E2E profiles.
func ModelTypeOverrideFromEnv() string {
	if value := normalizeRuntimeValue(os.Getenv(embeddingModelTypeOverrideEnv)); value != "" {
		return value
	}
	return normalizeRuntimeValue(os.Getenv("EMBEDDING_MODEL_OVERRIDE"))
}

// RuntimePlan is the effective text-embedding backend and model selected for
// both runtime preparation and classification execution.
type RuntimePlan struct {
	Backend       string
	ModelType     string
	ModelPath     string
	LocalOverride bool
}

// ResolveRuntimePlan applies backend/model overrides to the configured model
// contract. Switching a remote config to a local backend requires an explicit
// local model selection, or exactly one configured local model path.
func ResolveRuntimePlan(models config.EmbeddingModels, backendOverride string, modelOverride string) (RuntimePlan, error) {
	configuredBackend := models.EmbeddingBackend()
	backend := resolveRuntimeBackend(configuredBackend, backendOverride)
	switch backend {
	case config.EmbeddingBackendOpenAICompatible:
		return resolveRemoteRuntimePlan(models, backend), nil
	case config.EmbeddingBackendCandle, config.EmbeddingBackendOpenVINO:
		return resolveLocalRuntimePlan(models, configuredBackend, backend, modelOverride)
	default:
		return RuntimePlan{}, fmt.Errorf("unsupported embedding backend %q", backend)
	}
}

func resolveRuntimeBackend(configuredBackend string, backendOverride string) string {
	backend := normalizeRuntimeValue(backendOverride)
	if backend == "" {
		return configuredBackend
	}
	return backend
}

func resolveRemoteRuntimePlan(models config.EmbeddingModels, backend string) RuntimePlan {
	modelType := normalizeRuntimeValue(models.EmbeddingConfig.ModelType)
	if modelType == "" {
		modelType = config.EmbeddingModelTypeRemote
	}
	return RuntimePlan{Backend: backend, ModelType: modelType}
}

func resolveLocalRuntimePlan(
	models config.EmbeddingModels,
	configuredBackend string,
	backend string,
	modelOverride string,
) (RuntimePlan, error) {
	localOverride := configuredBackend == config.EmbeddingBackendOpenAICompatible
	modelType, modelPath, supported, err := resolveLocalModel(models, modelOverride)
	if err != nil {
		return RuntimePlan{}, err
	}
	if err := validateLocalRuntimePlan(backend, modelType, modelPath, supported, localOverride); err != nil {
		return RuntimePlan{}, err
	}
	return RuntimePlan{Backend: backend, ModelType: modelType, ModelPath: modelPath, LocalOverride: localOverride}, nil
}

func resolveLocalModel(models config.EmbeddingModels, modelOverride string) (string, string, bool, error) {
	modelType := normalizeRuntimeValue(modelOverride)
	if modelType == "" {
		modelType = normalizeRuntimeValue(models.EmbeddingConfig.ModelType)
	}
	if modelType == "" {
		modelType = config.EmbeddingModelTypeQwen3
	}
	if modelType == config.EmbeddingModelTypeRemote {
		inferredType, inferredPath, err := inferOnlyConfiguredLocalModel(models)
		if err != nil {
			return "", "", false, err
		}
		return inferredType, inferredPath, true, nil
	}
	modelType = canonicalLocalModelType(modelType)
	modelPath, supported := configuredLocalModelPath(models, modelType)
	return modelType, modelPath, supported, nil
}

func validateLocalRuntimePlan(backend string, modelType string, modelPath string, supported bool, localOverride bool) error {
	if backend == config.EmbeddingBackendOpenVINO && modelType == "multimodal" {
		return fmt.Errorf("embedding backend %q does not support model type %q", backend, modelType)
	}
	if !supported {
		return fmt.Errorf("local embedding backend %q does not support model type %q", backend, modelType)
	}
	if localOverride && strings.TrimSpace(modelPath) == "" {
		return fmt.Errorf("local embedding override requires a configured %s model path", modelType)
	}
	return nil
}

func inferOnlyConfiguredLocalModel(models config.EmbeddingModels) (string, string, error) {
	type candidate struct {
		modelType string
		path      string
	}
	candidates := make([]candidate, 0, 4)
	for _, item := range []candidate{
		{modelType: config.EmbeddingModelTypeQwen3, path: models.Qwen3ModelPath},
		{modelType: "gemma", path: models.GemmaModelPath},
		{modelType: "mmbert", path: models.MmBertModelPath},
		{modelType: "multimodal", path: models.MultiModalModelPath},
	} {
		if strings.TrimSpace(item.path) != "" {
			candidates = append(candidates, item)
		}
	}
	if len(candidates) == 0 {
		return "", "", fmt.Errorf("local embedding override requires an explicit local model and model path")
	}
	if len(candidates) > 1 {
		return "", "", fmt.Errorf("local embedding override is ambiguous across %d configured models; set %s", len(candidates), embeddingModelTypeOverrideEnv)
	}
	return candidates[0].modelType, candidates[0].path, nil
}

func configuredLocalModelPath(models config.EmbeddingModels, modelType string) (string, bool) {
	switch canonicalLocalModelType(modelType) {
	case config.EmbeddingModelTypeQwen3:
		return models.Qwen3ModelPath, true
	case "gemma":
		return models.GemmaModelPath, true
	case "mmbert":
		return models.MmBertModelPath, true
	case "multimodal":
		return models.MultiModalModelPath, true
	default:
		return "", false
	}
}

func canonicalLocalModelType(modelType string) string {
	modelType = normalizeRuntimeValue(modelType)
	if modelType == "modernbert" {
		return "mmbert"
	}
	return modelType
}

func normalizeRuntimeValue(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}
