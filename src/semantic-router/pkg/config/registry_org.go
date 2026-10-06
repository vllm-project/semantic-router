package config

import "strings"

// Hugging Face moved the project's repositories from this organization to
// vllm-sr with their commits intact, and redirects the old IDs.
const legacyHuggingFaceOrg = "llm-semantic-router"

// SameModelRepo reports whether two Hugging Face repo IDs name one repository,
// so mom_registry entries recorded before the organization move still match the
// registry.
func SameModelRepo(a, b string) bool {
	return canonicalModelRepo(a) == canonicalModelRepo(b)
}

func canonicalModelRepo(repoID string) string {
	if org, name, ok := strings.Cut(repoID, "/"); ok && strings.EqualFold(org, legacyHuggingFaceOrg) {
		return "vllm-sr/" + name
	}
	return repoID
}
