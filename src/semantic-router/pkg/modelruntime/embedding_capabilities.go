package modelruntime

import (
	"context"
	"fmt"
	"slices"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func remoteEmbeddingRequirement(requirement config.EmbeddingRequirement) error {
	if requirement.Windows || (requirement.Modality != "" && requirement.Modality != "text") || (requirement.Layer > 0 && !requirement.LocalLayerHint) {
		return fmt.Errorf("%w: %s requires local tokenizer/layer/modality capabilities; the HTTP embedding contract provides text vectors only (use explicit bindings for independent local consumers)", binding.ErrCapability, requirement.Consumer)
	}
	return nil
}

func validatePreparedEmbeddings(ctx context.Context, requirements []config.EmbeddingRequirement, providers map[string]embedding.Provider) error {
	checked := map[string]bool{}
	for _, requirement := range requirements {
		provider := providers[requirement.Model]
		if provider == nil {
			return fmt.Errorf("%w: %s has no prepared %s embedding", binding.ErrCapability, requirement.Consumer, requirement.Model)
		}
		options := embedding.Options{Dimension: requirement.Dimension, Layer: requirement.Layer}
		if provider.Backend() == config.EmbeddingBackendOpenAICompatible {
			if err := remoteEmbeddingRequirement(requirement); err != nil {
				return err
			}
			if requirement.LocalLayerHint {
				options.Layer = 0
			}
		}
		if requirement.Modality != "" && requirement.Modality != "text" {
			info, ok := provider.(embedding.Described)
			if !ok || !slices.Contains(info.EmbeddingInfo().Modalities, requirement.Modality) {
				return fmt.Errorf("%w: %s requires an actual %s encoder", binding.ErrCapability, requirement.Consumer, requirement.Modality)
			}
		}
		if requirement.Windows {
			windows, ok := provider.(embedding.WindowProvider)
			if !ok {
				return fmt.Errorf("%w: %s requires tokenizer windows", binding.ErrCapability, requirement.Consumer)
			}
			if _, err := windows.Windows(ctx, "semantic router capability probe", 0); err != nil {
				return fmt.Errorf("prepare %s tokenizer windows: %w", requirement.Consumer, err)
			}
		}
		key := fmt.Sprintf("%s:%d:%d", embedding.Identity(provider), options.Dimension, options.Layer)
		if checked[key] || options == (embedding.Options{}) {
			continue
		}
		if _, err := embedding.Embed(ctx, provider, "semantic router capability probe", options); err != nil {
			return fmt.Errorf("prepare %s vector semantics: %w", requirement.Consumer, err)
		}
		checked[key] = true
	}
	return nil
}

func batchedEmbeddingNeeds(cfg *config.RouterConfig, paths embeddingPaths) (bool, bool, error) {
	semanticCacheNeedsBatched := false
	semanticCacheModelType := resolveSemanticCacheEmbeddingModel(cfg)
	if cfg.Enabled && unifiedEmbeddingModelConfigured(paths, semanticCacheModelType) {
		capabilities, err := candle_binding.EmbeddingCapabilitiesFor(semanticCacheModelType)
		if err != nil {
			return false, false, fmt.Errorf("semantic cache embedding capabilities: %w", err)
		}
		semanticCacheNeedsBatched = capabilities.SupportsBatching
	}

	mlSelectionNeedsBatched := false
	if cfg.ModelSelection.Enabled &&
		cfg.ModelSelection.ML.ModelsPath != "" {
		mlModelType := strings.TrimSpace(cfg.ModelSelection.ML.ModelType)
		if mlModelType == "" {
			mlModelType = string(candle_binding.DefaultEmbeddingModelType)
		}
		if !unifiedEmbeddingModelConfigured(paths, mlModelType) {
			return semanticCacheNeedsBatched, false, nil
		}
		capabilities, err := candle_binding.EmbeddingCapabilitiesFor(mlModelType)
		if err != nil {
			return false, false, fmt.Errorf("ML selection embedding capabilities: %w", err)
		}
		mlSelectionNeedsBatched = capabilities.SupportsBatching
	}
	return semanticCacheNeedsBatched, mlSelectionNeedsBatched, nil
}

func unifiedEmbeddingModelConfigured(paths embeddingPaths, modelType string) bool {
	switch strings.ToLower(strings.TrimSpace(modelType)) {
	case "qwen3":
		return paths.qwen3 != ""
	case "gemma":
		return paths.gemma != ""
	case "mmbert":
		return paths.mmBert != ""
	default:
		return false
	}
}
