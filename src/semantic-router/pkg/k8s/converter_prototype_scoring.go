package k8s

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func convertPrototypeScoring(source *v1alpha1.PrototypeScoringConfig) *config.PrototypeScoringConfig {
	if source == nil {
		return nil
	}
	copy := source.DeepCopy()
	return &config.PrototypeScoringConfig{
		Enabled:                    copy.Enabled,
		ClusterSimilarityThreshold: copy.ClusterSimilarityThreshold,
		MaxPrototypes:              copy.MaxPrototypes,
		BestWeight:                 copy.BestWeight,
		TopM:                       copy.TopM,
		MarginThreshold:            copy.MarginThreshold,
	}
}
