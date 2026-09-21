package config

import "testing"

func TestVelaHaluDefaultAndLegacyIdentity(t *testing.T) {
	cfg := DefaultGlobalConfig()
	if got := cfg.HallucinationMitigation.HallucinationModel.ModelID; got != "models/Vela-1.0-Encoder-307M-Halu" {
		t.Fatalf("Halu default = %q", got)
	}
	model := GetModelByPath(cfg.HallucinationMitigation.HallucinationModel.ModelID)
	if model == nil || model.DefaultProvider != "candle" || model.DefaultDevice != "cpu" || model.DefaultAdapter != "vela_halu" || model.MaxContextLength != 8192 || len(model.Revision) != 40 {
		t.Fatalf("invalid Halu task identity: %+v", model)
	}
	legacy := GetModelByPath("models/mom-halugate-detector")
	if legacy == nil || legacy.RepoID != "KRLabsOrg/lettucedect-base-modernbert-en-v1" || legacy.DefaultAdapter != "" {
		t.Fatalf("legacy detector silently retargeted: %+v", legacy)
	}
	policy := cfg.HallucinationMitigation.HallucinationModel
	if policy.Threshold != .5 || policy.MinSpanLength != 1 || policy.MinSpanConfidence != 0 || policy.EnableNLIFiltering {
		t.Fatalf("default postprocessing changes published Halu verdict: %+v", policy)
	}
}

func TestVelaHaluBindingTaskBudget(t *testing.T) {
	decl := ModelBinding{Deployment: "halu", Adapter: "vela_halu", Contract: RemoteClassifierContractTokenSpans}
	for _, provider := range []string{"candle", "ort"} {
		deployment := ModelDeployment{Artifact: "models/custom-halu", Provider: provider, Device: "cpu", Input: ModelInputBudget{MaxTokens: 8192, Overflow: "reject"}}
		if err := validateTaskModelBinding("hallucination_detector", decl, deployment); err != nil {
			t.Fatal(err)
		}
		deployment.Input.MaxTokens = 32768
		if err := validateTaskModelBinding("hallucination_detector", decl, deployment); err == nil {
			t.Fatal("Halu inherited encoder's unqualified 32K capacity")
		}
	}
	if err := validateTaskModelBinding("pii_classifier", decl, ModelDeployment{Provider: "candle"}); err == nil {
		t.Fatal("Halu accepted an incompatible text-only input task")
	}
}
