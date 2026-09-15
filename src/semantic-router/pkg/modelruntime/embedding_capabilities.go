package modelruntime

import (
	"context"
	"fmt"
	"slices"

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
