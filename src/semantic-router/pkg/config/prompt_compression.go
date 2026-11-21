package config

import (
	"fmt"
	"slices"
	"strings"
)

const (
	PromptCompressionProfileDefault   = "default"
	PromptCompressionProfileCoding    = "coding"
	PromptCompressionProfileMedical   = "medical"
	PromptCompressionProfileSecurity  = "security"
	PromptCompressionProfileMultiTurn = "multi_turn"
)

var promptCompressionProfiles = []string{
	PromptCompressionProfileDefault,
	PromptCompressionProfileCoding,
	PromptCompressionProfileMedical,
	PromptCompressionProfileSecurity,
	PromptCompressionProfileMultiTurn,
}

// PromptCompressionConfig is a context plugin's compress.signals: it
// compresses the text the routing signals read, before they run, to the
// sentences its profile scores highest. The signals SkipSignals names read the
// original text.
type PromptCompressionConfig struct {
	Enabled        bool     `json:"enabled" yaml:"enabled"`
	Profile        string   `json:"profile,omitempty" yaml:"profile,omitempty"`
	MaxTokens      int      `json:"max_tokens" yaml:"max_tokens"`
	MinLength      int      `json:"min_length,omitempty" yaml:"min_length,omitempty"`
	SkipSignals    []string `json:"skip_signals,omitempty" yaml:"skip_signals,omitempty"`
	TextRankWeight float64  `json:"textrank_weight,omitempty" yaml:"textrank_weight,omitempty"`
	PositionWeight float64  `json:"position_weight,omitempty" yaml:"position_weight,omitempty"`
	TFIDFWeight    float64  `json:"tfidf_weight,omitempty" yaml:"tfidf_weight,omitempty"`
	NoveltyWeight  float64  `json:"novelty_weight,omitempty" yaml:"novelty_weight,omitempty"`
	PositionDepth  float64  `json:"position_depth,omitempty" yaml:"position_depth,omitempty"`
	PreserveFirstN int      `json:"preserve_first_n,omitempty" yaml:"preserve_first_n,omitempty"`
	PreserveLastN  int      `json:"preserve_last_n,omitempty" yaml:"preserve_last_n,omitempty"`
}

func (pc PromptCompressionConfig) SkipSignalsSet() map[string]bool {
	signals := pc.SkipSignals
	if len(signals) == 0 {
		signals = []string{SignalTypeJailbreak, SignalTypePII}
	}
	m := make(map[string]bool, len(signals))
	for _, s := range signals {
		m[s] = true
	}
	return m
}

// ValidPromptCompressionProfiles returns the stable built-in profile names
// accepted by the canonical config contract.
func ValidPromptCompressionProfiles() []string {
	return append([]string(nil), promptCompressionProfiles...)
}

// NormalizePromptCompressionProfile canonicalizes user-facing profile aliases.
func NormalizePromptCompressionProfile(profile string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(profile)), "-", "_")
}

// NormalizedProfile returns the configured profile or default when omitted.
func (pc PromptCompressionConfig) NormalizedProfile() string {
	profile := NormalizePromptCompressionProfile(pc.Profile)
	if profile == "" {
		return PromptCompressionProfileDefault
	}
	return profile
}

// validateSignalsCompression checks the compress.signals block at path.
func validateSignalsCompression(path string, signals *PromptCompressionConfig) error {
	if !slices.Contains(promptCompressionProfiles, signals.NormalizedProfile()) {
		return fmt.Errorf("%s.profile: unknown profile %q (valid profiles: %s)",
			path, signals.Profile, strings.Join(promptCompressionProfiles, ", "))
	}
	if signals.Enabled && signals.MaxTokens < 1 {
		return fmt.Errorf("%s.max_tokens must be at least 1: the signals read the text compressed down to it", path)
	}
	return nil
}
