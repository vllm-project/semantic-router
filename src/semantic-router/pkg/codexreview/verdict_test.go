package codexreview

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func answer(texts ...string) *llmprotocol.Response {
	response := &llmprotocol.Response{}
	for _, text := range texts {
		response.Output = append(response.Output, llmprotocol.OutputItem{
			Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
		})
	}
	return response
}

func TestVerdict(t *testing.T) {
	reasoning := llmprotocol.OutputItem{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentReasoning, Text: "**Applying strict JSON output schema**"}}}
	withReasoning := answer(`{"outcome":"allow"}`)
	withReasoning.Output = append([]llmprotocol.OutputItem{reasoning}, withReasoning.Output...)
	toolCall := &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentText, Text: "Checking the file first."},
		{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c1", Name: "exec_command", Arguments: `{"cmd":"ls"}`}},
	}}}}
	for name, test := range map[string]struct {
		response   *llmprotocol.Response
		status     string
		assessment *store.ReviewAssessment
	}{
		"allow only, as Codex asks for low risk": {answer(`{"outcome":"allow"}`), "completed", &store.ReviewAssessment{Outcome: "allow"}},
		"deny with every field": {
			answer(" {\"risk_level\":\"high\",\"user_authorization\":\"unknown\",\"outcome\":\"deny\",\"rationale\":\" Deletes the home directory. \"}\n"),
			"completed", &store.ReviewAssessment{Outcome: "deny", RiskLevel: "high", UserAuthorization: "unknown", Rationale: "Deletes the home directory."},
		},
		"reasoning before the answer": {withReasoning, "completed", &store.ReviewAssessment{Outcome: "allow"}},
		"the last message answers":    {answer("Let me think.", `{"outcome":"deny","risk_level":"critical"}`), "completed", &store.ReviewAssessment{Outcome: "deny", RiskLevel: "critical"}},
		"invalid JSON":                {answer(`{"outcome":"allow"`), "invalid", nil},
		"JSON in a code fence":        {answer("```json\n{\"outcome\":\"allow\"}\n```"), "invalid", nil},
		"missing outcome":             {answer(`{"risk_level":"low"}`), "invalid", nil},
		"outcome outside the schema":  {answer(`{"outcome":"escalate"}`), "invalid", nil},
		"risk outside the schema":     {answer(`{"outcome":"allow","risk_level":"tiny"}`), "invalid", nil},
		"authorization not a string":  {answer(`{"outcome":"allow","user_authorization":3}`), "invalid", nil},
		"plain text":                  {answer("I would allow this."), "invalid", nil},
		"no answer":                   {&llmprotocol.Response{}, "invalid", nil},
		"no response":                 {nil, "invalid", nil},
		"tool check before deciding":  {toolCall, "tool_call", nil},
		"upstream error": {
			&llmprotocol.Response{Error: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "server_error", "overloaded", nil)},
			"failed", nil,
		},
	} {
		t.Run(name, func(t *testing.T) {
			status, assessment := Verdict(test.response)
			if status != test.status {
				t.Fatalf("status = %q, want %q", status, test.status)
			}
			if (assessment == nil) != (test.assessment == nil) || assessment != nil && *assessment != *test.assessment {
				t.Fatalf("assessment = %+v, want %+v", assessment, test.assessment)
			}
		})
	}
}

// Verdict keeps the rationale whole for redaction; BoundRationale cuts it on
// a rune boundary afterwards.
func TestVerdictKeepsTheRationaleForRedactionThenBoundsIt(t *testing.T) {
	long := strings.Repeat("위험", 2000)
	status, assessment := Verdict(answer(`{"outcome":"deny","rationale":"` + long + `"}`))
	if status != "completed" || assessment.Rationale != long {
		t.Fatalf("status %q, rationale of %d bytes", status, len(assessment.Rationale))
	}
	bounded, truncated := BoundRationale(assessment.Rationale)
	if !truncated || len(bounded) > maxRationaleBytes || !strings.HasSuffix(bounded, "험") {
		t.Fatalf("rationale of %d bytes, truncated %v", len(bounded), truncated)
	}
}
