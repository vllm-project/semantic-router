package config

import (
	"fmt"
	"math"
)

// The three verdicts a complexity rule can reach. They are matched downstream
// as "<rule>:<verdict>", so the vocabulary is part of the routing contract and
// not a local implementation detail.
const (
	ComplexityDifficultyHard   = "hard"
	ComplexityDifficultyEasy   = "easy"
	ComplexityDifficultyMedium = "medium"
)

// ComplexityBoundaries are the two scores that separate a rule's three
// verdicts. They are resolved from a rule rather than read directly, because a
// rule may state them as an explicit pair or as the symmetric `threshold`
// shorthand.
//
// HardAt and EasyAt are in whatever units the score arrives in: the signed
// margin for local prototype scoring, the model's own units for a score.v1
// backend. HigherIsHarder records which way difficulty runs, so the same
// comparison serves both directions.
type ComplexityBoundaries struct {
	HardAt         float64
	EasyAt         float64
	HigherIsHarder bool
}

// Verdict classifies a score against the boundaries, returning the same
// hard/easy/medium vocabulary the local path has always produced. The band
// between the boundaries is medium: the score is not far enough either way to
// commit.
func (b ComplexityBoundaries) Verdict(score float64) string {
	if b.HigherIsHarder {
		switch {
		case score > b.HardAt:
			return ComplexityDifficultyHard
		case score < b.EasyAt:
			return ComplexityDifficultyEasy
		}
		return ComplexityDifficultyMedium
	}
	switch {
	case score < b.HardAt:
		return ComplexityDifficultyHard
	case score > b.EasyAt:
		return ComplexityDifficultyEasy
	}
	return ComplexityDifficultyMedium
}

// EffectiveBoundaries resolves a rule's declared boundaries.
//
// `threshold: X` remains the symmetric shorthand it has always been, because
// the local margin is signed and centred on zero. An explicit pair names its
// own direction: hard_above/easy_below where a higher score is harder,
// hard_below/easy_above where a lower one is. Encoding direction in the field
// names keeps a separate `direction` setting out of the schema and makes an
// overlapping band impossible to write by accident.
func (r ComplexityRule) EffectiveBoundaries() (ComplexityBoundaries, error) {
	for name, value := range map[string]*float64{
		"hard_above": r.HardAbove,
		"easy_below": r.EasyBelow,
		"hard_below": r.HardBelow,
		"easy_above": r.EasyAbove,
	} {
		// A non-finite cut point cannot separate anything: every comparison
		// against NaN is false, so the rule would answer medium for every
		// score, and an infinity makes one verdict unreachable.
		if value != nil && (math.IsNaN(*value) || math.IsInf(*value, 0)) {
			return ComplexityBoundaries{}, fmt.Errorf(
				"complexity rule %q has a non-finite %s (%v); a boundary must be a finite number",
				r.Name, name, *value)
		}
	}

	higher := r.HardAbove != nil || r.EasyBelow != nil
	lower := r.HardBelow != nil || r.EasyAbove != nil

	switch {
	case higher && lower:
		return ComplexityBoundaries{}, fmt.Errorf(
			"complexity rule %q states two directions at once: use hard_above with easy_below, or hard_below with easy_above",
			r.Name)
	case higher:
		if r.Threshold != 0 {
			return ComplexityBoundaries{}, fmt.Errorf(
				"complexity rule %q sets both threshold and an explicit boundary pair; keep one", r.Name)
		}
		if r.HardAbove == nil || r.EasyBelow == nil {
			return ComplexityBoundaries{}, fmt.Errorf(
				"complexity rule %q declares half a boundary pair; hard_above and easy_below are both required",
				r.Name)
		}
		if *r.EasyBelow >= *r.HardAbove {
			return ComplexityBoundaries{}, fmt.Errorf(
				"complexity rule %q has overlapping bands: easy_below (%v) must be below hard_above (%v)",
				r.Name, *r.EasyBelow, *r.HardAbove)
		}
		return ComplexityBoundaries{HardAt: *r.HardAbove, EasyAt: *r.EasyBelow, HigherIsHarder: true}, nil
	case lower:
		if r.Threshold != 0 {
			return ComplexityBoundaries{}, fmt.Errorf(
				"complexity rule %q sets both threshold and an explicit boundary pair; keep one", r.Name)
		}
		if r.HardBelow == nil || r.EasyAbove == nil {
			return ComplexityBoundaries{}, fmt.Errorf(
				"complexity rule %q declares half a boundary pair; hard_below and easy_above are both required",
				r.Name)
		}
		if *r.HardBelow >= *r.EasyAbove {
			return ComplexityBoundaries{}, fmt.Errorf(
				"complexity rule %q has overlapping bands: hard_below (%v) must be below easy_above (%v)",
				r.Name, *r.HardBelow, *r.EasyAbove)
		}
		return ComplexityBoundaries{HardAt: *r.HardBelow, EasyAt: *r.EasyAbove, HigherIsHarder: false}, nil
	}

	threshold := float64(r.Threshold)
	return ComplexityBoundaries{HardAt: threshold, EasyAt: -threshold, HigherIsHarder: true}, nil
}

// declaresBoundaryPair reports whether a rule states its cut points
// explicitly, rather than relying on the symmetric threshold shorthand.
func (r ComplexityRule) declaresBoundaryPair() bool {
	return r.HardAbove != nil || r.EasyBelow != nil || r.HardBelow != nil || r.EasyAbove != nil
}
