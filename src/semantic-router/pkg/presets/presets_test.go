package presets

import (
	"strings"
	"testing"
)

// Every preset is a valid System One question within the protocol limits,
// with unique option keys, and names only its own options as positive.
func TestCatalogQuestionsAreValid(t *testing.T) {
	seen := map[string]bool{}
	for _, preset := range catalog {
		key := preset.ID + "/" + preset.Stage
		if seen[key] {
			t.Fatalf("%s: declared twice", key)
		}
		seen[key] = true
		if preset.Instructions == "" || (preset.Stage != StageRequest && preset.Stage != StageResponse) {
			t.Fatalf("%s: instructions and a stage are required", key)
		}
		keys := map[string]bool{}
		for _, option := range preset.Options {
			if option.Key == "" || option.Description == "" || keys[option.Key] {
				t.Fatalf("%s: option %+v", key, option)
			}
			keys[option.Key] = true
		}
		switch preset.Type {
		case Choice:
			if preset.ID != ToolRank && (len(preset.Options) < MinChoices || len(preset.Options) > MaxChoices) {
				t.Fatalf("%s: %d choices", key, len(preset.Options))
			}
		case Set, Span:
			if len(preset.Options) < MinLabels || len(preset.Options) > MaxLabels {
				t.Fatalf("%s: %d labels", key, len(preset.Options))
			}
		case Noul:
			if len(preset.Options) > 0 {
				t.Fatalf("%s: a noul has no options", key)
			}
		default:
			t.Fatalf("%s: type %q", key, preset.Type)
		}
		for _, positive := range preset.Positive {
			if !keys[positive] {
				t.Fatalf("%s: positive %q is not an option", key, positive)
			}
		}
		if preset.RequiresSpan != (preset.Type == Span) {
			t.Fatalf("%s: span questions, and only they, require span answers", key)
		}
		if preset.Input.Answer != (preset.Stage == StageResponse) {
			t.Fatalf("%s: a response-stage question reads the answer, a request-stage one cannot", key)
		}
	}
}

// A trained binding answers canonical options: its trained options map to
// options of the preset, and what it covers is among them.
func TestTrainedBindingsAnswerCanonicalOptions(t *testing.T) {
	for key, binding := range bindings {
		preset, ok := Lookup(key.preset, key.stage)
		if !ok {
			t.Fatalf("%+v: binds a preset the catalog does not define", key)
		}
		// The Router's own IDs always contain a ':', so they never equal a trained one.
		if strings.Contains(binding.ID, ":") {
			t.Fatalf("%+v: trained ID %q", key, binding.ID)
		}
		for _, option := range binding.Options {
			if !preset.HasOption(binding.CanonicalKey(option.Key)) {
				t.Fatalf("%+v: trained option %q answers no canonical option", key, option.Key)
			}
		}
		for _, covered := range binding.Covers {
			if !preset.HasOption(covered) {
				t.Fatalf("%+v: covers %q, which is not an option", key, covered)
			}
		}
		if (binding.NativeSet || binding.LabelNouls != nil) && preset.Type != Set {
			t.Fatalf("%+v: a set binding on a %s preset", key, preset.Type)
		}
		if binding.ModelPreset != "" && preset.Type != Span {
			t.Fatalf("%+v: a model preset stands in for span presets only", key)
		}
	}
	modality, _ := Trained(FamilyVela3D3, Modality, StageRequest)
	if !modality.CoversOptions([]string{"image", "text_image"}) || modality.CoversOptions([]string{"image", "speech"}) {
		t.Fatal("the trained modality question answers text, image and text_image only")
	}
	if _, ok := Trained("decision3", Domain, StageRequest); ok {
		t.Fatal("the d3 family answers every preset zero-shot")
	}
}

func TestCalibratedThresholdsArePresetsOfTheCatalog(t *testing.T) {
	for model, thresholds := range calibration {
		for preset, threshold := range thresholds {
			if _, ok := Lookup(preset, StageRequest); !ok || threshold <= 0 || threshold >= 1 {
				t.Fatalf("%s/%s: %v", model, preset, threshold)
			}
		}
	}
	if threshold, ok := Threshold("vllm-sr/vela3-fast", Jailbreak, StageRequest); !ok || threshold != 0.88 {
		t.Fatalf("vela3-fast jailbreak: %v %v", threshold, ok)
	}
	if _, ok := Threshold("vllm-sr/vela3-fast", Jailbreak, StageResponse); ok {
		t.Fatal("thresholds were calibrated for the request stage only")
	}
	if _, ok := Threshold("vllm-sr/vela3-lite", Domain, StageRequest); ok {
		t.Fatal("vela3-lite has no calibrated thresholds")
	}
}
