package config

import (
	"os"
	"strings"
)

// DefaultEmbeddingExecution normalizes the legacy selector into the same
// deployment contract as explicit bindings. Explicit bindings bypass defaults.
func DefaultEmbeddingExecution(models EmbeddingModels) (provider, device string) {
	selected := models.EmbeddingBackend()
	if override := strings.TrimSpace(os.Getenv("EMBEDDING_BACKEND_OVERRIDE")); override != "" {
		selected = strings.ToLower(override)
	}
	if selected == EmbeddingBackendOpenVINO {
		return openvinoDefaultExecution(models.UseCPU)
	}
	return DefaultModelExecution(models.UseCPU)
}

// DefaultCategoryExecution preserves the legacy category selector at preparation
// time; request execution always uses the resolved owned handle.
func DefaultCategoryExecution(useCPU bool) (provider, device string) {
	if strings.EqualFold(strings.TrimSpace(os.Getenv("EMBEDDING_BACKEND_OVERRIDE")), EmbeddingBackendOpenVINO) {
		return openvinoDefaultExecution(useCPU)
	}
	return DefaultModelExecution(useCPU)
}

func openvinoDefaultExecution(useCPU bool) (string, string) {
	if useCPU {
		return "openvino", "CPU"
	}
	return "openvino", "AUTO"
}
