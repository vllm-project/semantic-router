package classification

import (
	"context"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// MaxImageFeatureCandidates caps total decoding work per request.
const MaxImageFeatureCandidates = 4

func (c *Classifier) recordImageFeatureEvidence(ctx context.Context, images []string, results *SignalResults) {
	for _, score := range c.Config.Projections.Scores {
		for _, input := range score.Inputs {
			if input.Weight != 0 && strings.EqualFold(input.Type, config.ProjectionInputImageFeatures) {
				writeImageFeatureEvidence(ctx, images, results)
				return
			}
		}
	}
}

func writeImageFeatureEvidence(ctx context.Context, images []string, results *SignalResults) {
	var maxima imageFeatures
	available := false
	for _, url := range images[:min(len(images), MaxImageFeatureCandidates)] {
		if ctx.Err() != nil {
			break
		}
		if features, ok := extractInlineImageFeatures(url); ok {
			available = true
			maxima.blur = max(maxima.blur, features.blur)
			maxima.edgeDensity = max(maxima.edgeDensity, features.edgeDensity)
		}
	}
	for name, value := range map[string]float64{"blur": maxima.blur, "edge_density": maxima.edgeDensity} {
		key := signalConfidenceKey(config.ProjectionInputImageFeatures, name)
		if available && ctx.Err() == nil {
			results.SignalValues[key] = value
		} else {
			results.SignalErrors[key] = "image_features_unavailable"
		}
	}
}
