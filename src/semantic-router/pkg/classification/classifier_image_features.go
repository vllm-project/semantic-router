package classification

import (
	"context"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (c *Classifier) recordImageFeatureEvidence(ctx context.Context, images []string, results *SignalResults, mu *sync.Mutex) {
	for _, score := range c.Config.Projections.Scores {
		for _, input := range score.Inputs {
			if input.Weight != 0 && strings.EqualFold(input.Type, config.ProjectionInputImageFeatures) {
				writeImageFeatureEvidence(ctx, images, results, mu)
				return
			}
		}
	}
}

func writeImageFeatureEvidence(ctx context.Context, images []string, results *SignalResults, mu *sync.Mutex) {
	maxima, available := aggregateImageFeatures(ctx, images)
	mu.Lock()
	defer mu.Unlock()
	available = available && ctx.Err() == nil
	for name, value := range map[string]float64{"blur": maxima.blur, "edge_density": maxima.edgeDensity} {
		key := signalConfidenceKey(config.ProjectionInputImageFeatures, name)
		if available {
			results.SignalValues[key] = value
		} else {
			results.SignalErrors[key] = "image_features_unavailable"
		}
	}
}
