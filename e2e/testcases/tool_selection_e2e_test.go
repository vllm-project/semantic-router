package testcases

import (
	"encoding/json"
	"testing"
)

func TestToolSelectionRequestsUseOwnedEntrypoints(t *testing.T) {
	cases := toolSelectionContractCases(json.RawMessage(`{"type":"object","properties":{}}`))
	if len(cases) != 10 {
		t.Fatalf("tool contract inventory = %d, want 10", len(cases))
	}
	var featureCases, precedenceCases int
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			req := buildToolSelectionChatRequest(tc)
			model := "e2e-plugins"
			if tc.Name == "pii_decision_runs_before_tool_selection" {
				model = "MoM"
				precedenceCases++
			} else {
				featureCases++
			}
			if req.Model != model {
				t.Fatalf("request model = %q, want %q", req.Model, model)
			}
			if len(req.Messages) != 1 || req.Messages[0].Content != tc.Prompt || req.Messages[0].Role != "user" {
				t.Fatal("entrypoint selection changed the authored user request")
			}
		})
	}
	if featureCases != 9 || precedenceCases != 1 {
		t.Fatalf("feature/precedence inventory = %d/%d, want 9/1", featureCases, precedenceCases)
	}
}
