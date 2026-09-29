package testcases

import (
	"net/http"
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

func TestValidateLooperResponsesSynthesisStream(t *testing.T) {
	body := []byte(`event: response.created
data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_fixture","status":"in_progress"}}

event: response.in_progress
data: {"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_fixture","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_fixture","type":"message","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","sequence_number":3,"output_index":0,"item_id":"msg_fixture","content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":4,"output_index":0,"item_id":"msg_fixture","content_index":0,"delta":"fusion-none-answer"}

event: response.output_text.done
data: {"type":"response.output_text.done","sequence_number":5,"output_index":0,"item_id":"msg_fixture","content_index":0,"text":"fusion-none-answer"}

event: response.content_part.done
data: {"type":"response.content_part.done","sequence_number":6,"output_index":0,"item_id":"msg_fixture","content_index":0,"part":{"type":"output_text","text":"fusion-none-answer","annotations":[]}}

event: response.output_item.done
data: {"type":"response.output_item.done","sequence_number":7,"output_index":0,"item":{"id":"msg_fixture","type":"message","role":"assistant","content":[{"type":"output_text","text":"fusion-none-answer","annotations":[]}]}}

event: response.completed
data: {"type":"response.completed","sequence_number":8,"response":{"id":"resp_fixture","status":"completed","output":[{"id":"msg_fixture","type":"message","role":"assistant","content":[{"type":"output_text","text":"fusion-none-answer","annotations":[]}]}]}}

`)
	valid := responseAPIStreamingSSEResult{statusCode: http.StatusOK, contentType: "text/event-stream", body: body, protocolWarnings: "dropped;router_extension_unsupported_protocol;fusion"}
	for _, name := range []string{"complete", "buffered_200", "missing_completion", "missing_diagnostic", "wrong_answer"} {
		t.Run(name, func(t *testing.T) {
			result := valid
			switch name {
			case "buffered_200":
				result.contentType = "application/json"
			case "missing_completion":
				result.body = []byte(strings.ReplaceAll(string(body), "event: response.completed", "event: incomplete"))
			case "missing_diagnostic":
				result.protocolWarnings = ""
			case "wrong_answer":
				result.body = []byte(strings.ReplaceAll(string(body), "fusion-none-answer", "panel-answer"))
			}
			if err := validateLooperResponsesSynthesisStream(result); (err == nil) != (name == "complete") {
				t.Fatalf("validation=%v for %s", err, name)
			}
		})
	}
}
