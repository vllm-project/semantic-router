package config

import (
	"fmt"
	"math"
)

// Model scores are static 0-1 weights: the selectors compare them directly
// and the Elo conversion maps them onto its rating range. A non-finite or
// out-of-band score has no interpretation at selection time, so the load
// path rejects it here instead of letting it reach the selector loop.
func validateCategoryModelScores(category Category) error {
	for _, modelScore := range category.ModelScores {
		if err := validateModelScore(category.Name, modelScore); err != nil {
			return err
		}
	}
	return nil
}

func validateModelScore(categoryName string, modelScore ModelScore) error {
	score := modelScore.Score
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return fmt.Errorf(
			"routing.signals.domains[%q].model_scores[%q].score must be finite, got %v",
			categoryName,
			modelScore.Model,
			score,
		)
	}
	if score < 0 || score > 1 {
		return fmt.Errorf(
			"routing.signals.domains[%q].model_scores[%q].score must be between 0.0 and 1.0, got %v",
			categoryName,
			modelScore.Model,
			score,
		)
	}
	return nil
}
