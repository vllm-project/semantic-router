package config

import (
	"fmt"
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

func validatePromptCompressionContracts(cfg *RouterConfig) error {
	profile := cfg.PromptCompression.NormalizedProfile()
	for _, valid := range promptCompressionProfiles {
		if profile == valid {
			return nil
		}
	}
	return fmt.Errorf(
		"unknown prompt_compression.profile %q (valid profiles: %s)",
		cfg.PromptCompression.Profile,
		strings.Join(promptCompressionProfiles, ", "),
	)
}
