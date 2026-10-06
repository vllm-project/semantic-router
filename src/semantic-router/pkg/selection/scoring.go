package selection

import (
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ScoreComponent supplies higher-is-better utilities for a weighted comparison.
// NormalizeCandidateScores converts raw higher/lower-is-better measurements
// when normalization is part of the algorithm. Ordinal ranks are not utilities.
type ScoreComponent struct {
	Weight float64
	Scores CandidateScores
}

func NormalizeCandidateScores(scores CandidateScores, direction ScoreDirection) CandidateScores {
	result := append(CandidateScores(nil), scores...)
	minimum, maximum := math.Inf(1), math.Inf(-1)
	for _, row := range result {
		minimum, maximum = math.Min(minimum, row.Score), math.Max(maximum, row.Score)
	}
	for i := range result {
		value := normalizeDirect(result[i].Score, minimum, maximum, true)
		if direction == LowerIsBetter {
			value = 1 - value
		}
		result[i].Score = value
	}
	return result
}

// BlendCandidateScores preserves Hybrid's pool-wide available-component weight
// normalization. A missing component contributes no value, not benchmark evidence.
// Caller order fixes floating-point accumulation and final unresolved tie order.
func BlendCandidateScores(candidates []config.ModelRef, components []ScoreComponent) CandidateScores {
	total := 0.0
	for _, component := range components {
		if len(component.Scores) > 0 {
			total += component.Weight
		}
	}
	result := make(CandidateScores, len(candidates))
	for i, candidate := range candidates {
		result[i].Candidate = candidate
		if total == 0 {
			result[i].Score = 1.0 / float64(len(candidates))
			continue
		}
		for _, component := range components {
			if score, available := component.Scores.Get(candidate); available {
				result[i].Score += component.Weight / total * score
			}
		}
	}
	return result
}
