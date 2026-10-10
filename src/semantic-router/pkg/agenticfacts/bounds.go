package agenticfacts

import (
	"time"
)

// Bounds caps every dimension of an envelope that a caller controls. Each field
// exists because the proposal requires validation to bound size, depth,
// cardinality, and lifetime before facts influence selection.
//
// An unset field is filled from DefaultBounds rather than treated as unlimited,
// so a forgotten field cannot silently disable a guard. ClockSkew is the one
// exception: zero skew is a deliberate choice, not an omission.
type Bounds struct {
	MaxEnvelopeBytes int
	MaxDepth         int
	MaxCapabilities  int
	MaxStringLength  int
	MaxLifetime      time.Duration
	ClockSkew        time.Duration
}

// DefaultBounds returns the caps applied when an operator declares none. The
// values are deliberately tight: the envelope carries identifiers and labels,
// never content, so a well-formed payload has no reason to approach them.
func DefaultBounds() Bounds {
	return Bounds{
		MaxEnvelopeBytes: 8192,
		MaxDepth:         16,
		MaxCapabilities:  16,
		MaxStringLength:  128,
		MaxLifetime:      5 * time.Minute,
		ClockSkew:        5 * time.Second,
	}
}

// withDefaults substitutes DefaultBounds values for any unset field, so a zero
// Bounds validates as strictly as DefaultBounds rather than not at all.
func (b Bounds) withDefaults() Bounds {
	defaults := DefaultBounds()
	if b.MaxEnvelopeBytes <= 0 {
		b.MaxEnvelopeBytes = defaults.MaxEnvelopeBytes
	}
	if b.MaxDepth <= 0 {
		b.MaxDepth = defaults.MaxDepth
	}
	if b.MaxCapabilities <= 0 {
		b.MaxCapabilities = defaults.MaxCapabilities
	}
	if b.MaxStringLength <= 0 {
		b.MaxStringLength = defaults.MaxStringLength
	}
	if b.MaxLifetime <= 0 {
		b.MaxLifetime = defaults.MaxLifetime
	}
	if b.ClockSkew < 0 {
		b.ClockSkew = defaults.ClockSkew
	}
	return b
}
