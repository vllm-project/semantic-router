package classification

import (
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// ValidateImageClassification checks a prepared recipe without calling a model.
// Provider wrappers can expose EmbedImage even when their model is text-only,
// so both the actual capabilities and image-query rules must be present.
func (c *Classifier) ValidateImageClassification() error {
	if c == nil || c.keywordEmbeddingClassifier == nil || len(c.keywordEmbeddingClassifier.rulesByModality[config.QueryModalityImage]) == 0 {
		return fmt.Errorf("%w: intent image classification requires prepared image embedding rules", binding.ErrCapability)
	}
	provider := c.keywordEmbeddingClassifier.provider
	described, ok := provider.(embedding.Described)
	if !ok || !slices.Contains(described.EmbeddingInfo().Modalities, "image") {
		return fmt.Errorf("%w: intent image classification requires a prepared image encoder", binding.ErrCapability)
	}
	if _, ok := provider.(embedding.ImageProvider); !ok {
		return fmt.Errorf("%w: prepared embedding provider cannot encode images", binding.ErrCapability)
	}
	return nil
}
