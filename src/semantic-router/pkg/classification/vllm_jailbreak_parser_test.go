package classification

import "testing"

func TestVLLMJailbreakParserPreservesVerdictsWithoutProbabilities(t *testing.T) {
	for _, tt := range []struct{ parser, output, label string }{
		{"qwen3guard", "Safety: Unsafe\nCategories: Jailbreak, Illegal", "unsafe"},
		{"qwen3guard", "Severity level: controversial\nCategories: None", "controversial"},
		{"qwen3guard", "Safety: Safe\nCategories: None", "safe"},
		{"auto", `{"is_jailbreak":true}`, "unsafe"},
		{"json", `{"is_unsafe":false}`, "safe"},
		{"simple", "This content is unsafe.", "unsafe"},
	} {
		t.Run(tt.parser+tt.label, func(t *testing.T) {
			v := &VLLMJailbreakInference{parserType: tt.parser}
			decision, err := v.parseSafetyOutput(tt.output)
			if err != nil || decision.Label != tt.label {
				t.Fatalf("decision=%+v error=%v", decision, err)
			}
			if decision.Score != nil || decision.ScoreSemantics != nil {
				t.Fatal("parser invented a score")
			}
		})
	}
}

func TestVLLMJailbreakParserRejectsMissingOrAmbiguousVerdicts(t *testing.T) {
	for _, text := range []string{"", "no structured result", "Categories: harmful behavior", "not unsafe", `{"is_unsafe":false} trailing`, "the quoted jailbreak example is safe"} {
		if decision, err := (&VLLMJailbreakInference{}).parseSafetyOutput(text); err == nil {
			t.Fatalf("accepted %q: %+v", text, decision)
		}
	}
}
