package config

import "strings"

// Semantic-cache polarity guard modes.
//
// The lexical tier (#2691) is the unconditional, model-free floor: it is never
// switched off by this setting. The NLI tier (#2751) is opt-in and verifies the
// single best above-threshold candidate with the shared hallucination explainer
// NLI model before it is served.
const (
	PolarityGuardModeLexical    = "lexical"
	PolarityGuardModeNLI        = "nli"
	PolarityGuardModeLexicalNLI = "lexical+nli"

	// DefaultPolarityContradictionThreshold is the NLI contradiction probability
	// above which a cache candidate is rejected.
	DefaultPolarityContradictionThreshold float32 = 0.5
)

// PolarityGuardConfig configures the semantic-cache negation/antonym guard.
type PolarityGuardConfig struct {
	// Mode selects the guard tiers: "lexical" (default), "nli", or "lexical+nli".
	Mode string `yaml:"mode,omitempty"`
	// NLI tunes the optional NLI tier.
	NLI PolarityGuardNLIConfig `yaml:"nli,omitempty"`
}

// PolarityGuardNLIConfig tunes the NLI polarity tier.
//
// The tier does not bind its own model: the native binding holds exactly one
// NLI model, so it reuses the hallucination explainer configured under
// global.model_catalog.modules.hallucination_mitigation.explainer.
type PolarityGuardNLIConfig struct {
	// ContradictionThreshold rejects the candidate when the contradiction
	// probability exceeds it. Defaults to DefaultPolarityContradictionThreshold.
	ContradictionThreshold *float32 `yaml:"contradiction_threshold,omitempty"`
}

// NormalizedMode returns the trimmed, lower-cased mode, defaulting to lexical.
// It is nil-safe so an absent polarity_guard block reads as the default.
func (c *PolarityGuardConfig) NormalizedMode() string {
	if c == nil {
		return PolarityGuardModeLexical
	}
	mode := strings.ToLower(strings.TrimSpace(c.Mode))
	if mode == "" {
		return PolarityGuardModeLexical
	}
	return mode
}

// UsesNLI reports whether the NLI tier is enabled by the configured mode.
func (c *PolarityGuardConfig) UsesNLI() bool {
	switch c.NormalizedMode() {
	case PolarityGuardModeNLI, PolarityGuardModeLexicalNLI:
		return true
	default:
		return false
	}
}

// EffectiveContradictionThreshold returns the configured NLI contradiction
// threshold or the default when unset.
func (c *PolarityGuardConfig) EffectiveContradictionThreshold() float32 {
	if c == nil || c.NLI.ContradictionThreshold == nil {
		return DefaultPolarityContradictionThreshold
	}
	return *c.NLI.ContradictionThreshold
}

// polarityGuardModeSupported reports whether mode is a known guard mode.
func polarityGuardModeSupported(mode string) bool {
	switch mode {
	case PolarityGuardModeLexical, PolarityGuardModeNLI, PolarityGuardModeLexicalNLI:
		return true
	default:
		return false
	}
}
