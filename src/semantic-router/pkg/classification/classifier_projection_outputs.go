package classification

import (
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// applyProjectionMapping records the matched output band(s) for a mapping
// according to its method. threshold_bands and the empty default emit only the
// first matching band; multi_emit emits every matching band so orthogonal policy
// tags can propagate simultaneously.
func applyProjectionMapping(
	mapping config.ProjectionMapping,
	scoreValue float64,
	results *SignalResults,
) {
	switch mapping.Method {
	case config.ProjectionMappingMethodMultiEmit:
		for _, output := range mapping.Outputs {
			if projectionOutputMatches(output, scoreValue) {
				recordProjectionMatch(mapping, output, scoreValue, results)
			}
		}
	default:
		if output, matched := matchProjectionOutput(mapping, scoreValue); matched {
			recordProjectionMatch(mapping, output, scoreValue, results)
		}
	}
}

func recordProjectionMatch(
	mapping config.ProjectionMapping,
	output config.ProjectionMappingOutput,
	scoreValue float64,
	results *SignalResults,
) {
	results.MatchedProjectionRules = append(results.MatchedProjectionRules, output.Name)
	results.SignalConfidences[signalConfidenceKey(config.SignalTypeProjection, output.Name)] = projectionOutputConfidence(mapping, output, scoreValue)
}

func matchProjectionOutput(
	mapping config.ProjectionMapping,
	scoreValue float64,
) (config.ProjectionMappingOutput, bool) {
	for _, output := range mapping.Outputs {
		if projectionOutputMatches(output, scoreValue) {
			return output, true
		}
	}
	return config.ProjectionMappingOutput{}, false
}

func projectionOutputMatches(output config.ProjectionMappingOutput, scoreValue float64) bool {
	if output.GT != nil && !(scoreValue > *output.GT) {
		return false
	}
	if output.GTE != nil && !(scoreValue >= *output.GTE) {
		return false
	}
	if output.LT != nil && !(scoreValue < *output.LT) {
		return false
	}
	if output.LTE != nil && !(scoreValue <= *output.LTE) {
		return false
	}
	return true
}

func projectionOutputConfidence(
	mapping config.ProjectionMapping,
	output config.ProjectionMappingOutput,
	scoreValue float64,
) float64 {
	slope := 12.0
	if mapping.Calibration != nil && mapping.Calibration.Slope > 0 {
		slope = mapping.Calibration.Slope
	}

	distance := projectionBoundaryDistance(output, scoreValue)
	return 1.0 / (1.0 + math.Exp(-slope*distance))
}

func projectionBoundaryDistance(output config.ProjectionMappingOutput, scoreValue float64) float64 {
	distances := make([]float64, 0, 4)
	if output.GT != nil {
		distances = append(distances, math.Abs(scoreValue-*output.GT))
	}
	if output.GTE != nil {
		distances = append(distances, math.Abs(scoreValue-*output.GTE))
	}
	if output.LT != nil {
		distances = append(distances, math.Abs(*output.LT-scoreValue))
	}
	if output.LTE != nil {
		distances = append(distances, math.Abs(*output.LTE-scoreValue))
	}
	if len(distances) == 0 {
		return 1.0
	}

	best := distances[0]
	for _, distance := range distances[1:] {
		if distance < best {
			best = distance
		}
	}
	return best
}
