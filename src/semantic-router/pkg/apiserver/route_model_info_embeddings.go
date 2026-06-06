//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"path"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modellifecycle"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

// getEmbeddingModelsInfo returns information about loaded embedding models.
func (s *ClassificationAPIServer) getEmbeddingModelsInfo(runtimeState *startupstatus.State) []ModelInfo {
	var models []ModelInfo

	embeddingInfo, err := candle_binding.GetEmbeddingModelsInfo()
	if err != nil {
		logging.Warnf("Failed to get embedding models info: %v", err)
	} else {
		for _, model := range embeddingInfo.Models {
			modelPath := normalizeEmbeddingModelPath(model.ModelPath, model.ModelName)
			if modelPath == "" {
				modelPath = strings.TrimSpace(model.ModelPath)
			}
			if modelPath == "" {
				modelPath = strings.TrimSpace(model.ModelName)
			}

			models = append(models, ModelInfo{
				Name:      fmt.Sprintf("%s_embedding_model", model.ModelName),
				Type:      "embedding",
				Loaded:    model.IsLoaded,
				ModelPath: modelPath,
				Metadata: map[string]string{
					"model_type":           model.ModelName,
					"max_sequence_length":  fmt.Sprintf("%d", model.MaxSequenceLength),
					"default_dimension":    fmt.Sprintf("%d", model.DefaultDimension),
					"matryoshka_supported": "true",
				},
			})
		}
	}

	models = appendConfiguredEmbeddingModelPlaceholders(models, s.currentConfig())
	for i := range models {
		models[i] = enrichModelInfo(models[i], runtimeState)
	}

	return models
}

func appendConfiguredEmbeddingModelPlaceholders(models []ModelInfo, cfg *routerconfig.RouterConfig) []ModelInfo {
	if cfg == nil {
		return models
	}

	seen := configuredEmbeddingModelPathSet(models)
	plan := modellifecycle.BuildPlan(cfg)
	for _, candidate := range configuredEmbeddingModelCandidates(plan) {
		if candidate.path == "" {
			continue
		}
		canonicalPath := canonicalModelPath(candidate.path)
		if canonicalPath == "" {
			canonicalPath = candidate.path
		}
		if _, ok := seen[canonicalPath]; ok {
			continue
		}
		seen[canonicalPath] = struct{}{}
		models = append(models, ModelInfo{
			Name:      candidate.name,
			Type:      "embedding",
			Loaded:    false,
			ModelPath: canonicalPath,
			Metadata: map[string]string{
				"model_type": candidate.modelType,
				"source":     "model_lifecycle_plan",
			},
		})
	}
	return models
}

func configuredEmbeddingModelPathSet(models []ModelInfo) map[string]struct{} {
	seen := make(map[string]struct{}, len(models))
	for _, model := range models {
		modelPath := canonicalModelPath(model.ModelPath)
		if modelPath == "" {
			modelPath = model.ModelPath
		}
		if modelPath != "" {
			seen[modelPath] = struct{}{}
		}
	}
	return seen
}

type embeddingModelCandidate struct {
	name      string
	modelType string
	path      string
}

func configuredEmbeddingModelCandidates(plan modellifecycle.Plan) []embeddingModelCandidate {
	return []embeddingModelCandidate{
		embeddingModelCandidateForRole(plan, modellifecycle.RoleQwen3Embedding, "qwen3"),
		embeddingModelCandidateForRole(plan, modellifecycle.RoleGemmaEmbedding, "gemma"),
		embeddingModelCandidateForRole(plan, modellifecycle.RoleMmBERTEmbedding, "mmbert"),
		embeddingModelCandidateForRole(plan, modellifecycle.RoleMultiModalEmbedding, "multimodal"),
		embeddingModelCandidateForRole(plan, modellifecycle.RoleBERTEmbedding, "bert"),
	}
}

func embeddingModelCandidateForRole(
	plan modellifecycle.Plan,
	role modellifecycle.AssetRole,
	modelType string,
) embeddingModelCandidate {
	asset, _ := plan.AssetForRole(role)
	return embeddingModelCandidate{
		name:      fmt.Sprintf("%s_embedding_model", modelType),
		modelType: modelType,
		path:      asset.LocalPath,
	}
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
