package testcases

import (
	"strings"
	"testing"
)

func TestValidateLooperSynthesisTrace(t *testing.T) {
	const valid = `{"object":"chat.completion","choices":[{"message":{"content":"fusion-none-answer"}}],"fusion":{"responses":[{"model":"fusion-panel-valid"}],"failed_models":[{"model":"fusion-panel-fail"}]}}`
	for _, tc := range []struct {
		name, body string
		valid      bool
	}{
		{"complete", valid, true},
		{"trace_omitted", `{"object":"chat.completion","choices":[{"message":{"content":"fusion-none-answer"}}]}`, false},
		{"failure_evidence_lost", strings.Replace(valid, `[{"model":"fusion-panel-fail"}]`, `[]`, 1), false},
		{"panel_answer_only", strings.Replace(valid, "fusion-none-answer", "usable-fusion-panel-answer", 1), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateLooperSynthesisTrace([]byte(tc.body))
			if (err == nil) != tc.valid {
				t.Fatalf("validation error=%v, want valid=%t", err, tc.valid)
			}
		})
	}
}
