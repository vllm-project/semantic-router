package modelruntime

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// validatePreparedEmbeddings embeds once at every view a consumer reads, so a
// deployment that cannot serve it fails preparation, not a request.
func validatePreparedEmbeddings(ctx context.Context, requirements []config.EmbeddingRequirement, providers map[string]embedding.Provider) error {
	checked := map[string]bool{}
	for _, requirement := range requirements {
		provider := providers[requirement.Deployment]
		if provider == nil {
			return fmt.Errorf("%w: %s has no prepared embedding deployment %s", binding.ErrCapability, requirement.Consumer, requirement.Deployment)
		}
		options := embedding.Options{Dimension: requirement.Dimension, Layer: requirement.Layer}
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
