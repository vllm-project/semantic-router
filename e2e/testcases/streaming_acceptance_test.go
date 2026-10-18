package testcases

import (
	"net/http"
	"strings"
	"testing"
)

func TestStreamingSafetyRejectsUnrelatedFastResponses(t *testing.T) {
	for _, tc := range []struct {
		name, decision, fast, body, wantDecision, refusal string
		wantError                                         bool
	}{
		{"refusal", "block_pii", "true", `{"choices":[{"message":{"content":"denied"}}]}`, "block_pii", "denied", false},
		{"other blocker", "block_jailbreak", "true", `{"choices":[{"message":{"content":"denied"}}]}`, "block_pii", "denied", true},
		{"wrong refusal", "block_pii", "true", `{"choices":[{"message":{"content":"unrelated"}}]}`, "block_pii", "denied", true},
		{"no decision", "", "true", `{"choices":[{"message":{"content":"denied"}}]}`, "block_pii", "denied", true},
		{"invalid body", "block_pii", "true", `not json`, "block_pii", "denied", true},
		{"no fast response", "block_pii", "", `{"choices":[{"message":{"content":"denied"}}]}`, "block_pii", "denied", true},
		{"benign control", "general_decision", "", `{"choices":[{"message":{"content":"answer"}}]}`, "general_decision", "", false},
		{"universal blocking", "general_decision", "true", `{"choices":[{"message":{"content":"denied"}}]}`, "general_decision", "", true},
		{"empty control", "general_decision", "", `{"choices":[{"message":{"content":""}}]}`, "general_decision", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			headers := make(http.Header)
			headers.Set("x-vsr-selected-decision", tc.decision)
			headers.Set("x-vsr-fast-response", tc.fast)
			err := validateStreamingSafetyResponse(headers, []byte(tc.body), tc.wantDecision, tc.refusal)
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError = %v", err, tc.wantError)
			}
		})
	}
}

func TestStreamingCacheRequiresActualSSEReplay(t *testing.T) {
	const replay = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	const done = "data: [DONE]\n\n"
	const finish = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
	const content = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}\n\n"
	const usage = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n"
	for _, tc := range []struct {
		name, hit, contentType, body string
		wantError                    bool
	}{
		{"complete replay", "true", "text/event-stream; charset=utf-8", replay, false},
		{"usage after finish", "true", "text/event-stream", strings.Replace(replay, done, usage+done, 1), false},
		{"content after finish", "true", "text/event-stream", finish + content + done, true},
		{"duplicate finish", "true", "text/event-stream", strings.Replace(replay, done, finish+done, 1), true},
		{"error finish overwritten", "true", "text/event-stream", strings.Replace(finish, "stop", "error", 1) + replay, true},
		{"empty choice after finish", "true", "text/event-stream", strings.Replace(replay, done, strings.Replace(finish, "\"stop\"", "null", 1)+done, 1), true},
		{"not cached", "false", "text/event-stream", replay, true},
		{"JSON only", "true", "application/json", `{"choices":[]}`, true},
		{"empty stream", "true", "text/event-stream", "", true},
		{"truncated stream", "true", "text/event-stream", strings.ReplaceAll(replay, "data: [DONE]\n\n", ""), true},
		{"wrong content", "true", "text/event-stream", strings.ReplaceAll(replay, "answer", "other"), true},
		{"extra content", "true", "text/event-stream", strings.ReplaceAll(replay, "answer", "answer extra"), true},
		{"duplicate terminal", "true", "text/event-stream", replay + "data: [DONE]\n\n", true},
		{"early terminal", "true", "text/event-stream", "data: [DONE]\n\n" + strings.ReplaceAll(replay, "data: [DONE]\n\n", ""), true},
		{"error terminal", "true", "text/event-stream", strings.ReplaceAll(replay, "stop", "error"), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			headers := make(http.Header)
			headers.Set("x-vsr-cache-hit", tc.hit)
			headers.Set("Content-Type", tc.contentType)
			err := validateStreamingCacheReplay(headers, []byte(tc.body), "answer")
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError = %v", err, tc.wantError)
			}
		})
	}
}
