package config

import (
	"strings"
	"testing"
)

func appliedModuleThresholds(cfg *RouterConfig) ModuleThresholds {
	return ModuleThresholds{
		PromptGuard: cfg.PromptGuard.Threshold, Domain: cfg.CategoryModel.Threshold, PII: cfg.PIIModel.Threshold,
		FactCheck: cfg.HallucinationMitigation.FactCheckModel.Threshold, Feedback: cfg.FeedbackDetector.Threshold,
	}
}

func uncalibratedFields(cfg *RouterConfig) []string {
	var fields []string
	for _, warning := range Warnings(cfg) {
		if warning.Code == "uncalibrated_preset" {
			fields = append(fields, warning.Field)
		}
	}
	return fields
}

// A module that writes no threshold takes the one its answering model is
// calibrated for. A model without calibration (a d3 size, a Vela 3.0 d3 size,
// a Decision 2.0 model) never inherits another model's: it takes 0.5, and the
// load names every such module with an uncalibrated_preset warning.
func TestModuleThresholdsComeFromTheAnsweringModel(t *testing.T) {
	cases := []struct {
		artifact string
		want     ModuleThresholds
		warns    bool
	}{
		{"vllm-sr/vela3-fast", ModuleThresholdsOf(Vela3SignalModel), false},
		{"vllm-sr/Vela-2.0-4B", ModuleThresholdsOf(Vela2Model4B), false},
		{"vllm-sr/vela3-lite", neutralModuleThresholds, true},
		{"vllm-sr/d3-lite", neutralModuleThresholds, true},
		{"vllm-sr/Decision-2.0-Kai-0.6B", neutralModuleThresholds, true},
	}
	for _, tc := range cases {
		cfg := mustParseDecisionModel(t, selectedDecisionYAML(tc.artifact, ""))
		if got := appliedModuleThresholds(cfg); got != tc.want {
			t.Fatalf("%s: thresholds %+v, want %+v", tc.artifact, got, tc.want)
		}
		fields := uncalibratedFields(cfg)
		if tc.warns != (len(fields) == 5) || (!tc.warns && len(fields) > 0) {
			t.Fatalf("%s: uncalibrated_preset on %v", tc.artifact, fields)
		}
	}
}

// A Vela 1.0 specialist keeps its own thresholds, and a threshold the
// configuration writes is used as written, without a warning.
func TestSpecialistAndWrittenThresholdsAreCalibrated(t *testing.T) {
	cfg := mustParseDecisionModel(t, selectedDecisionYAML("vllm-sr/d3-lite", "      prompt_guard: models/Vela-1.0-Encoder-307M-Guard\n"))
	if got := cfg.PromptGuard.Threshold; got != vela1ModuleThresholds.PromptGuard {
		t.Fatalf("the Guard specialist's threshold: %v", got)
	}
	for _, field := range uncalibratedFields(cfg) {
		if strings.Contains(field, "prompt_guard") {
			t.Fatalf("a specialist is calibrated: %v", field)
		}
	}
	written := strings.Replace(selectedDecisionYAML("vllm-sr/d3-lite", ""), "    modules:\n", "    modules:\n      classifier:\n        pii: {threshold: 0.3}\n", 1)
	cfg = mustParseDecisionModel(t, written)
	if cfg.PIIModel.Threshold != 0.3 {
		t.Fatalf("a written threshold is used as written: %v", cfg.PIIModel.Threshold)
	}
	for _, field := range uncalibratedFields(cfg) {
		if strings.Contains(field, "classifier.pii") {
			t.Fatalf("a written threshold needs no calibration: %v", field)
		}
	}
}
