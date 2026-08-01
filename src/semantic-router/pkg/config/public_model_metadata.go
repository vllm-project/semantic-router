package config

import "strings"

const (
	defaultPublicModelOwner     = "vllm-semantic-router"
	amdPublicModelOwner         = "amd"
	defaultAutoModelDescription = "Intelligent Router for Mixture-of-Models"
	amdBalancedModelDescription = "AMD Mixture-of-Models · Balanced — quality, speed, and efficiency optimized for everyday routing."
)

// PublicModelOwner returns the catalog owner for a request-facing model.
// AMD-branded MoM IDs retain AMD ownership while other router aliases preserve
// the established semantic-router identity.
func PublicModelOwner(modelID string) string {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(modelID)), "amd/") {
		return amdPublicModelOwner
	}
	return defaultPublicModelOwner
}

// PublicAutoModelDescription returns the model-card description for an
// automatic routing alias.
func PublicAutoModelDescription(modelID string) string {
	normalized := strings.ToLower(strings.TrimSpace(modelID))
	if strings.HasPrefix(normalized, "amd/rocm-v1-balanced") {
		return amdBalancedModelDescription
	}
	return defaultAutoModelDescription
}

// PublicEntrypointDescription supplies product metadata for entrypoints whose
// recipe does not carry a description, notably the top-level default recipe.
func PublicEntrypointDescription(modelID, recipeName, recipeDescription string) string {
	if recipeName == DefaultRecipeName &&
		strings.HasPrefix(strings.ToLower(strings.TrimSpace(modelID)), "amd/rocm-v1-balanced") {
		return amdBalancedModelDescription
	}
	return recipeDescription
}
