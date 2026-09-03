package config

import (
	"strings"
	"testing"
)

func inputModalityTestConfig(rules ...InputModalityRule) *RouterConfig {
	return &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{InputModalityRules: rules},
		},
	}
}

func TestValidateInputModalityContractsAcceptsValidRules(t *testing.T) {
	cfg := inputModalityTestConfig(
		InputModalityRule{Name: "image_input", Modality: InputModalityImage},
		InputModalityRule{Name: "audio_input", Description: "audio present", Modality: InputModalityAudio},
	)
	if err := validateInputModalityContracts(cfg); err != nil {
		t.Fatalf("expected valid rules to pass, got %v", err)
	}
}

func TestValidateInputModalityContractsRejectsMissingName(t *testing.T) {
	cfg := inputModalityTestConfig(InputModalityRule{Modality: InputModalityImage})
	err := validateInputModalityContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), "name is required") {
		t.Fatalf("expected missing-name error, got %v", err)
	}
}

func TestValidateInputModalityContractsRejectsUnknownModality(t *testing.T) {
	cfg := inputModalityTestConfig(InputModalityRule{Name: "smell_input", Modality: "smell"})
	err := validateInputModalityContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), "modality must be one of") {
		t.Fatalf("expected unknown-modality error, got %v", err)
	}
}

func TestInputModalityForContentPartType(t *testing.T) {
	cases := map[string]struct {
		modality string
		ok       bool
	}{
		"text":        {InputModalityText, true},
		"input_text":  {InputModalityText, true},
		"image":       {"", false},
		"audio":       {"", false},
		"video":       {"", false},
		"image_url":   {InputModalityImage, true},
		"input_image": {InputModalityImage, true},
		"input_audio": {InputModalityAudio, true},
		"audio_url":   {InputModalityAudio, true},
		"video_url":   {InputModalityVideo, true},
		"input_video": {InputModalityVideo, true},
		"tool_use":    {"", false},
		"":            {"", false},
	}
	for partType, want := range cases {
		modality, ok := InputModalityForContentPartType(partType)
		if modality != want.modality || ok != want.ok {
			t.Errorf("InputModalityForContentPartType(%q) = (%q, %v), want (%q, %v)",
				partType, modality, ok, want.modality, want.ok)
		}
	}
	if modality, ok := InputModalityForContentPartType(" Image_URL "); !ok || modality != InputModalityImage {
		t.Errorf("InputModalityForContentPartType must trim and lowercase the type: got (%q, %v)", modality, ok)
	}
}

func TestValidateInputModalityContractsRejectsDuplicateNames(t *testing.T) {
	cfg := inputModalityTestConfig(
		InputModalityRule{Name: "image_input", Modality: InputModalityImage},
		InputModalityRule{Name: "Image_Input", Modality: InputModalityVideo},
	)
	err := validateInputModalityContracts(cfg)
	if err == nil || !strings.Contains(err.Error(), "duplicate name") {
		t.Fatalf("expected duplicate-name error, got %v", err)
	}
}
