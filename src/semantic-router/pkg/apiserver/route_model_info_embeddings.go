//go:build !windows && cgo

package apiserver

import (
	"path"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// getEmbeddingModelsInfo lists the current generation's prepared embedding
// bindings, including service and named-recipe owners of shared resources.
func (s *ClassificationAPIServer) getEmbeddingModelsInfo() []ModelInfo {
	_, service, release := s.acquireClassificationRuntime()
	defer release()
	models, _ := preparedEmbeddingModelsInfo(service)
	return models
}

func normalizeEmbeddingModelPath(runtimePath, modelName string) string {
	for _, candidate := range embeddingModelPathCandidates(runtimePath, modelName) {
		if spec := routerconfig.GetModelByPath(candidate); spec != nil {
			return spec.LocalPath
		}
	}

	return ""
}

func embeddingModelPathCandidates(values ...string) []string {
	seen := make(map[string]struct{})
	var candidates []string

	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}

		candidates = appendEmbeddingModelPathCandidate(candidates, seen, trimmed)

		extracted := extractEmbeddingModelPath(trimmed)
		if extracted != "" {
			candidates = appendEmbeddingModelPathCandidate(candidates, seen, extracted)
		}
	}

	return candidates
}

func appendEmbeddingModelPathCandidate(
	candidates []string,
	seen map[string]struct{},
	value string,
) []string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return candidates
	}

	if _, ok := seen[trimmed]; !ok {
		seen[trimmed] = struct{}{}
		candidates = append(candidates, trimmed)
	}

	base := path.Base(trimmed)
	if base == "." || base == "/" || base == trimmed {
		return candidates
	}

	if _, ok := seen[base]; !ok {
		seen[base] = struct{}{}
		candidates = append(candidates, base)
	}

	if !strings.HasPrefix(base, "models/") {
		modelsBase := "models/" + base
		if _, ok := seen[modelsBase]; !ok {
			seen[modelsBase] = struct{}{}
			candidates = append(candidates, modelsBase)
		}
	}

	return candidates
}

func extractEmbeddingModelPath(value string) string {
	if value == "" {
		return ""
	}

	const marker = "path="
	index := strings.Index(value, marker)
	if index == -1 {
		return ""
	}

	trimmed := value[index+len(marker):]
	if end := strings.IndexAny(trimmed, ",)"); end >= 0 {
		trimmed = trimmed[:end]
	}

	return strings.TrimSpace(trimmed)
}
