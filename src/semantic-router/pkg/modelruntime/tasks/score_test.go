package tasks

import (
	"math"
	"testing"
)

func TestScoreSemanticsPreserveRawUnits(t *testing.T) {
	semantics := ScoreSemantics{Unit: "difficulty", Direction: LowerIsPositive}
	for _, raw := range []float64{-15.5, 0, 1, 250} {
		result := ScoreResult{Value: raw}
		if err := semantics.Validate(result); err != nil {
			t.Errorf("valid raw score %v rejected: %v", raw, err)
		}
		if result.Value != raw || semantics.Calibrated {
			t.Fatal("validation changed units or claimed calibration")
		}
	}
	for _, raw := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if err := semantics.Validate(ScoreResult{Value: raw}); err == nil {
			t.Errorf("non-finite score %v accepted", raw)
		}
	}
}

func TestScoreSemanticsRejectRatherThanClamp(t *testing.T) {
	lower, upper := -10.0, 10.0
	semantics := ScoreSemantics{Unit: "difficulty", Direction: HigherIsPositive, Minimum: &lower, Maximum: &upper}
	for _, raw := range []float64{-11, 11} {
		result := ScoreResult{Value: raw}
		if err := semantics.Validate(result); err == nil || result.Value != raw {
			t.Errorf("out-of-range score was accepted or changed: %+v, %v", result, err)
		}
	}
}
