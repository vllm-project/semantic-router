package selection

import "time"

// Progress evidence version stamped into replay so calibration changes are
// auditable.
const ProgressEvidenceVersion = "pew-v1"

// TurnOutcomeFact is one content-minimal session-turn fact consumed by the
// switch gate. It mirrors sessiontelemetry.TurnOutcome without importing it,
// keeping this package free of storage dependencies.
type TurnOutcomeFact struct {
	TurnIndex         int
	Timestamp         time.Time
	Model             string
	Category          string
	ModelAttributable bool
	Confidence        float64
	ConfidenceKnown   bool
	OutputTokens      int64
	LatencyMs         int64
	LatencyKnown      bool
	Cost              float64
	CostKnown         bool
}

// Model-attributable outcome categories.
const (
	OutcomeProgress   = "progress"
	OutcomeNoProgress = "no_progress"
	OutcomeRegression = "regression"
)

// ProgressEvidence is the calibrated read of a bounded recent-outcome window.
// It is derived deterministically: no IO, no ambient clock.
type ProgressEvidence struct {
	Version string

	// Trend is the attributable outcome balance in [-1,1]; negative means the
	// window leans toward regression.
	Trend float64

	// RegressionStreak counts consecutive same-category attributable
	// regressions ending at the newest attributable outcome. RecoveryStreak
	// counts consecutive trailing progress outcomes.
	RegressionStreak int
	StreakCategory   string
	RecoveryStreak   int

	ConfidenceTrend      float64
	CostTrend            float64
	LatencyTrend         float64
	ConfidenceTrendKnown bool
	CostTrendKnown       bool
	LatencyTrendKnown    bool

	AttributableCount int
	MissingCount      int
	TotalCount        int

	ColdStart bool
}

// EvaluateProgressEvidence derives calibrated evidence from a recent window
// ordered oldest → newest. Non-attributable outcomes (provider/tool failures,
// missing turns) never break or extend a streak: they are environment noise,
// not model behaviour.
func EvaluateProgressEvidence(window []TurnOutcomeFact) ProgressEvidence {
	evidence := ProgressEvidence{
		Version:    ProgressEvidenceVersion,
		TotalCount: len(window),
		ColdStart:  len(window) == 0,
	}
	if len(window) == 0 {
		return evidence
	}

	var balance float64
	for _, fact := range window {
		if fact.Category == "missing" {
			evidence.MissingCount++
		}
		if !fact.ModelAttributable {
			continue
		}
		evidence.AttributableCount++
		switch fact.Category {
		case OutcomeProgress:
			balance++
		case OutcomeNoProgress:
			balance--
		case OutcomeRegression:
			balance -= 2
		}
	}
	if evidence.AttributableCount == 0 {
		evidence.ColdStart = true
		return evidence
	}

	evidence.Trend = clampUnit(balance / float64(2*evidence.AttributableCount))
	evidence.RegressionStreak, evidence.StreakCategory = trailingRegressionStreak(window)
	evidence.RecoveryStreak = trailingRecoveryStreak(window)
	evidence.ConfidenceTrend, evidence.ConfidenceTrendKnown = attributableDelta(window, func(f TurnOutcomeFact) (float64, bool) { return f.Confidence, f.ConfidenceKnown })
	evidence.CostTrend, evidence.CostTrendKnown = attributableDelta(window, func(f TurnOutcomeFact) (float64, bool) { return f.Cost, f.CostKnown })
	evidence.LatencyTrend, evidence.LatencyTrendKnown = attributableDelta(window, func(f TurnOutcomeFact) (float64, bool) { return float64(f.LatencyMs), f.LatencyKnown })
	return evidence
}

// trailingRegressionStreak counts consecutive same-category regressions at the
// newest end of the window, skipping non-attributable noise.
func trailingRegressionStreak(window []TurnOutcomeFact) (int, string) {
	streak := 0
	category := ""
	for i := len(window) - 1; i >= 0; i-- {
		fact := window[i]
		if !fact.ModelAttributable {
			continue
		}
		if fact.Category != OutcomeNoProgress && fact.Category != OutcomeRegression {
			break
		}
		if category == "" {
			category = fact.Category
		} else if fact.Category != category {
			break
		}
		streak++
	}
	return streak, category
}

// trailingRecoveryStreak counts consecutive progress outcomes at the newest end
// of the window, skipping non-attributable noise.
func trailingRecoveryStreak(window []TurnOutcomeFact) int {
	streak := 0
	for i := len(window) - 1; i >= 0; i-- {
		fact := window[i]
		if !fact.ModelAttributable {
			continue
		}
		if fact.Category != OutcomeProgress {
			break
		}
		streak++
	}
	return streak
}

// attributableDelta reports newest minus oldest for an attributable metric,
// normalized by the oldest value so callers get a relative trend.
func attributableDelta(window []TurnOutcomeFact, value func(TurnOutcomeFact) (float64, bool)) (float64, bool) {
	var oldest, newest float64
	count := 0
	for _, fact := range window {
		v, known := value(fact)
		if !fact.ModelAttributable || !known {
			continue
		}
		if count == 0 {
			oldest = v
		}
		newest = v
		count++
	}
	if count < 2 || oldest == 0 {
		return 0, false
	}
	return (newest - oldest) / oldest, true
}

func clampUnit(v float64) float64 {
	if v > 1 {
		return 1
	}
	if v < -1 {
		return -1
	}
	return v
}
