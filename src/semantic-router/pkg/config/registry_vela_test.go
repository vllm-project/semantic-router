package config

import (
	"regexp"
	"slices"
	"testing"
)

func TestVelaReleaseRegistryContracts(t *testing.T) {
	immutableRevision := regexp.MustCompile(`^[0-9a-f]{40}$`)
	legacy := ToLegacyRegistry()
	for _, tc := range []struct {
		name    string
		purpose ModelPurpose
		classes int
	}{
		{"Vela-1.0-Encoder-307M", PurposeEncoder, 0},
		{"Vela-1.0-Encoder-307M-Domain", PurposeDomainClassification, 14},
		{"Vela-1.0-Encoder-307M-PII", PurposePIIDetection, 35},
		{"Vela-1.0-Encoder-307M-FactCheck", PurposeHallucinationSentinel, 2},
		{"Vela-1.0-Encoder-307M-Modality", PurposeModalityDetection, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := "models/" + tc.name
			repo := "llm-semantic-router/" + tc.name
			for _, alias := range []string{path, tc.name} {
				model := GetModelByPath(alias)
				if model == nil {
					t.Fatalf("release not found by %q", alias)
				}
				if model.LocalPath != path || model.RepoID != repo || ResolveModelPath(alias) != path || legacy[alias] != repo {
					t.Fatalf("inconsistent registry resolution for %q: %+v", alias, model)
				}
				if !immutableRevision.MatchString(model.Revision) {
					t.Fatalf("release must pin an immutable HF revision, got %q", model.Revision)
				}
				info := model.RegistryInfo()
				if info.Purpose != string(tc.purpose) || info.NumClasses != tc.classes || info.MaxContextLength != 32768 {
					t.Fatalf("incorrect task contract: %+v", info)
				}
				// The registered root artifact is self-contained even when the
				// same repository also publishes an optional lora/ variant.
				if info.UsesLoRA || info.Revision != model.Revision || !slices.Contains(info.Tags, "vela") {
					t.Fatalf("incorrect release metadata: %+v", info)
				}
			}
		})
	}
}
