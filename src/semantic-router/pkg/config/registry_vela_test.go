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
		{"Vela-1.0-Encoder-307M-Guard", PurposeJailbreakDetection, 2},
		{"Vela-1.0-Encoder-307M-Safety", PurposeSafety, 2},
		{"Vela-1.0-Encoder-307M-Shield", PurposeSafety, 2},
		{"Vela-1.0-Encoder-307M-Hazard", PurposeHazard, 12},
		{"Vela-1.0-Encoder-307M-Embedding", PurposeEmbedding, 0},
		{"Vela-1.0-Encoder-307M-Reranker", PurposeReranking, 0},
		{"Vela-1.0-Encoder-307M-Domain", PurposeDomainClassification, 14},
		{"Vela-1.0-Encoder-307M-PII", PurposePIIDetection, 35},
		{"Vela-1.0-Encoder-307M-FactCheck", PurposeHallucinationSentinel, 2},
		{"Vela-1.0-Encoder-307M-Modality", PurposeModalityDetection, 3},
		{"Vela-1.0-Encoder-307M-Feedback", PurposeFeedbackDetection, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := "models/" + tc.name
			repo := "vllm-sr/" + tc.name
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

func TestVelaShieldIsSelectableWithoutChangingTheSafetyDefault(t *testing.T) {
	if got := DefaultSystemModels().Safety; got != Vela2SignalModel {
		t.Fatalf("default safety model changed to %q", got)
	}
	if got := DefaultGlobalConfig().SafetyModels.Safety.ModelID; got != Vela2SignalModel {
		t.Fatalf("default safety module resolves to %q", got)
	}
	shield := GetModelByPath("models/Vela-1.0-Encoder-307M-Shield")
	if shield == nil || shield.Purpose != PurposeSafety {
		t.Fatalf("Shield is not registered as a safety model: %+v", shield)
	}
	for _, pattern := range velaTrainingArtifactPatterns {
		if !slices.Contains(shield.DownloadExcludePatterns, pattern) {
			t.Fatalf("Shield dropped the Vela training-artifact exclusion %q", pattern)
		}
	}
	for _, pattern := range []string{"lc/*", "heads/*", "demo.py", "DEMO_OUTPUT.txt"} {
		if !slices.Contains(shield.DownloadExcludePatterns, pattern) {
			t.Fatalf("Shield does not exclude %q", pattern)
		}
	}
}

func TestVela2SignalModelIsAPinnedRuntimeServedRelease(t *testing.T) {
	for _, alias := range []string{Vela2SignalModel, "Vela-2.0-0.3B"} {
		model := GetModelByPath(alias)
		if model == nil || model.LocalPath != Vela2SignalModel || model.RepoID != "vllm-sr/Vela-2.0-0.3B" {
			t.Fatalf("%q does not resolve to Vela 2.0 0.3B: %+v", alias, model)
		}
		if !regexp.MustCompile(`^[0-9a-f]{40}$`).MatchString(model.Revision) || !model.RuntimeProvisioned || !servedBuiltIn(model) {
			t.Fatalf("Vela 2.0 0.3B must be a pinned release the runtime serves: %+v", model)
		}
		if !model.SharedDeployment || model.CPUProfile != "max_speed" || model.MaxContextLength != 8192 {
			t.Fatalf("Vela 2.0 0.3B runs one shared deployment, max_speed on CPU, 8K input: %+v", model)
		}
	}
}
