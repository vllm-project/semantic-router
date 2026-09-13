package tasks

import (
	"fmt"
	"math"
)

// ScoreDirection describes how increasing values relate to a task's positive
// outcome. It does not turn a regression or similarity score into a probability.
type ScoreDirection string

const (
	HigherIsPositive ScoreDirection = "higher_is_positive"
	LowerIsPositive  ScoreDirection = "lower_is_positive"
)

// ScoreSemantics accompanies a scoring capability. Unit names the model's raw
// units; optional bounds describe its actual range, not a request to clamp it.
// Calibrated is true only when calibration is part of the adapter contract.
type ScoreSemantics struct {
	Unit       string
	Direction  ScoreDirection
	Minimum    *float64
	Maximum    *float64
	Calibrated bool
}

// ScoreResult carries a raw score without a fabricated label or probability.
type ScoreResult struct {
	Value float64
}

// Validate checks a score against known capability bounds without changing its
// value. Unbounded regression scores may be negative or greater than one.
func (s ScoreSemantics) Validate(result ScoreResult) error {
	if math.IsNaN(result.Value) || math.IsInf(result.Value, 0) {
		return fmt.Errorf("score must be finite")
	}
	if s.Minimum != nil && result.Value < *s.Minimum {
		return fmt.Errorf("score is below the declared minimum")
	}
	if s.Maximum != nil && result.Value > *s.Maximum {
		return fmt.Errorf("score is above the declared maximum")
	}
	return nil
}
