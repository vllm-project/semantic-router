//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"path"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

// getEmbeddingModelsInfo returns information about loaded embedding models.
func (s *ClassificationAPIServer) getEmbeddingModelsInfo(runtimeState *startupstatus.State) []ModelInfo {
	var models []ModelInfo

	prepared, release, err := s.acquireEmbeddings()
	if err != nil {
		return models
	}
	defer release()
	for _, model := range prepared.Models() {
		models = append(models, ModelInfo{Name: fmt.Sprintf("%s_embedding_model", model.Name), Type: "embedding", Loaded: true, ModelPath: model.Artifact, Metadata: map[string]string{"model_type": model.Name, "provider": model.Backend, "max_sequence_length": fmt.Sprint(model.MaxTokens), "default_dimension": fmt.Sprint(model.Dimension), "pooling": model.Pooling, "normalization": model.Normalization, "modalities": strings.Join(model.Modalities, ",")}})
	}

	for i := range models {
		models[i] = enrichModelInfo(models[i], runtimeState)
	}

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
