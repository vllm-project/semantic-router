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
	OutputTokens      int64
	LatencyMs         int64
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

	ConfidenceTrend float64
	CostTrend       float64
	LatencyTrend    float64

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
	evidence.ConfidenceTrend = attributableDelta(window, func(f TurnOutcomeFact) float64 { return f.Confidence })
	evidence.CostTrend = attributableDelta(window, func(f TurnOutcomeFact) float64 { return float64(f.OutputTokens) })
	evidence.LatencyTrend = attributableDelta(window, func(f TurnOutcomeFact) float64 { return float64(f.LatencyMs) })
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
func attributableDelta(window []TurnOutcomeFact, value func(TurnOutcomeFact) float64) float64 {
	first, last := -1, -1
	for i, fact := range window {
		if !fact.ModelAttributable {
			continue
		}
		if first < 0 {
			first = i
		}
		last = i
	}
	if first < 0 || first == last {
		return 0
	}
	oldest := value(window[first])
	newest := value(window[last])
	if oldest == 0 {
		return 0
	}
	return (newest - oldest) / oldest
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
