package config

import (
	"strings"
	"testing"
)

func TestValidateEmbeddingRuleModalities_AcceptsTextOnlyRulesWithAnyModelType(t *testing.T) {
	rules := []EmbeddingRule{
		{Name: "topic_text", Candidates: []string{"hello"}},
		{Name: "topic_text_explicit", Candidates: []string{"world"}, QueryModality: QueryModalityText},
	}
	for _, modelType := range []string{"qwen3", "gemma", "multimodal", ""} {
		if err := validateEmbeddingRuleModalities(rules, modelType); err != nil {
			t.Errorf("text-only rules should pass under model_type=%q, got: %v", modelType, err)
		}
	}
}

func TestValidateEmbeddingRuleModalities_RejectsImageRuleWithoutMultimodal(t *testing.T) {
	rules := []EmbeddingRule{
		{Name: "chip_fab_imagery", Candidates: []string{"wafer"}, QueryModality: QueryModalityImage},
	}
	err := validateEmbeddingRuleModalities(rules, "qwen3")
	if err == nil {
		t.Fatal("expected error for image rule paired with non-multimodal model_type, got nil")
	}
	msg := err.Error()
	for _, want := range []string{"chip_fab_imagery", "model_type=multimodal", "qwen3"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error should contain %q, got: %s", want, msg)
		}
	}
}

func TestValidateEmbeddingRuleModalities_AcceptsImageRuleUnderMultimodal(t *testing.T) {
	rules := []EmbeddingRule{
		{Name: "chip_fab_imagery", Candidates: []string{"wafer"}, QueryModality: QueryModalityImage},
	}
	if err := validateEmbeddingRuleModalities(rules, "multimodal"); err != nil {
		t.Errorf("image rule should pass under model_type=multimodal, got: %v", err)
	}
}

func TestValidateEmbeddingRuleModalities_RejectsAudioWithComingLaterMessage(t *testing.T) {
	rules := []EmbeddingRule{
		{Name: "rig_walkie_talkie_audio", Candidates: []string{"radio call"}, QueryModality: QueryModalityAudio},
	}
	err := validateEmbeddingRuleModalities(rules, "multimodal")
	if err == nil {
		t.Fatal("expected error for audio rule (FFI not yet wired), got nil")
	}
	msg := err.Error()
	for _, want := range []string{"rig_walkie_talkie_audio", "MultiModalEncodeAudioFromBase64", "planned"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error should contain %q, got: %s", want, msg)
		}
	}
}

func TestValidateEmbeddingRuleModalities_RejectsUnknownModality(t *testing.T) {
	rules := []EmbeddingRule{
		{Name: "typo_rule", Candidates: []string{"x"}, QueryModality: QueryModality("imag")},
	}
	err := validateEmbeddingRuleModalities(rules, "multimodal")
	if err == nil {
		t.Fatal("expected error for unknown query_modality, got nil")
	}
	msg := err.Error()
	for _, want := range []string{"typo_rule", "imag"} {
		if !strings.Contains(msg, want) {
			t.Errorf("error should contain %q, got: %s", want, msg)
		}
	}
}

func TestValidateEmbeddingRuleModalities_AggregatesAllProblems(t *testing.T) {
	rules := []EmbeddingRule{
		{Name: "img_rule", Candidates: []string{"x"}, QueryModality: QueryModalityImage},
		{Name: "audio_rule", Candidates: []string{"y"}, QueryModality: QueryModalityAudio},
		{Name: "typo_rule", Candidates: []string{"z"}, QueryModality: QueryModality("vidoe")},
	}
	err := validateEmbeddingRuleModalities(rules, "qwen3")
	if err == nil {
		t.Fatal("expected aggregated error, got nil")
	}
	msg := err.Error()
	for _, name := range []string{"img_rule", "audio_rule", "typo_rule"} {
		if !strings.Contains(msg, name) {
			t.Errorf("aggregated error should mention %q, got: %s", name, msg)
		}
	}
}

func TestValidateEmbeddingRuleModalities_NormalizesCaseAndWhitespace(t *testing.T) {
	rules := []EmbeddingRule{
		{Name: "image_uppercase", Candidates: []string{"x"}, QueryModality: QueryModality("  IMAGE  ")},
	}
	if err := validateEmbeddingRuleModalities(rules, "multimodal"); err != nil {
		t.Errorf("case-insensitive whitespace-trimmed image modality should pass, got: %v", err)
	}
}

func TestValidateEmbeddingContracts_NilConfigIsTolerant(t *testing.T) {
	if err := validateEmbeddingContracts(nil); err != nil {
		t.Errorf("nil config should not produce an error, got: %v", err)
	}
}

func TestValidateEmbeddingContracts_RoutesThroughTopLevelConfig(t *testing.T) {
	cfg := &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				EmbeddingRules: []EmbeddingRule{
					{Name: "img_rule", Candidates: []string{"x"}, QueryModality: QueryModalityImage},
				},
			},
		},
		InlineModels: InlineModels{
			EmbeddingModels: EmbeddingModels{
				EmbeddingConfig: HNSWConfig{ModelType: "qwen3"},
			},
		},
	}
	err := validateEmbeddingContracts(cfg)
	if err == nil {
		t.Fatal("expected validateEmbeddingContracts to surface the underlying validator error, got nil")
	}
	if !strings.Contains(err.Error(), "img_rule") {
		t.Errorf("error should name the offending rule, got: %s", err.Error())
	}
}

func TestEmbeddingRule_EffectiveQueryModalityDefaultsToText(t *testing.T) {
	cases := []struct {
		input QueryModality
		want  QueryModality
	}{
		{"", QueryModalityText},
		{"  ", QueryModalityText},
		{"text", QueryModalityText},
		{"TEXT", QueryModalityText},
		{"image", QueryModalityImage},
		{"  Image  ", QueryModalityImage},
		{"audio", QueryModalityAudio},
	}
	for _, tc := range cases {
		got := EmbeddingRule{QueryModality: tc.input}.EffectiveQueryModality()
		if got != tc.want {
			t.Errorf("EffectiveQueryModality(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
