package redaction

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestRedactResponseBodyRemovesTrajectoryContentAndArguments(t *testing.T) {
	body := []byte(`{
		"object":"router_replay.trajectory",
		"messages":[
			{"role":"user","content":"private prompt"},
			{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"secret\":true}"}}]},
			{"role":"tool","content":"private result","tool_call_id":"call-1"},
			{"role":"assistant","content":"private answer"}
		]
	}`)

	redacted, changed, err := RedactResponseBody(body)
	if err != nil {
		t.Fatalf("RedactResponseBody() error = %v", err)
	}
	if !changed {
		t.Fatal("RedactResponseBody() changed = false, want true")
	}

	var payload map[string]any
	if err := json.Unmarshal(redacted, &payload); err != nil {
		t.Fatalf("decode redacted trajectory: %v", err)
	}
	messages, ok := payload["messages"].([]any)
	if !ok || len(messages) != 4 {
		t.Fatalf("messages = %#v, want four entries", payload["messages"])
	}
	for index, rawMessage := range messages {
		message := rawMessage.(map[string]any)
		if content, ok := message["content"]; ok && content != "" {
			t.Fatalf("message %d content = %#v, want empty", index, content)
		}
	}
	toolCall := messages[1].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)
	function := toolCall["function"].(map[string]any)
	if function["arguments"] != "" {
		t.Fatalf("tool arguments = %#v, want empty", function["arguments"])
	}
	if messages[2].(map[string]any)["status"] != RedactedToolResultStatusSucceeded {
		t.Fatalf("tool result status = %#v, want %q", messages[2].(map[string]any)["status"], RedactedToolResultStatusSucceeded)
	}
}

func TestRedactResponseBodyRemovesRecordFreeTextAndPrivacySurfaces(t *testing.T) {
	const private = "private-canary"
	body := []byte(`{
		"id":"replay-1",
		"decision":"safe-decision",
		"request_body":"private-canary",
		"response_body":"private-canary",
		"prompt":"private-canary",
		"tool_definitions":"private-canary",
		"pii_entities":["private-canary"],
		"hallucination_spans":["private-canary"],
		"hallucination_span_details":[{"text":"private-canary","explanation":"private-canary","severity":2}],
		"outcomes":[{"source":"user","target":"model","target_ref":"private-canary","verdict":"failed","reason":"private-canary","metadata":{"note":"private-canary"}}],
		"session_policy":{"user_id":"private-canary"},
		"route_diagnostics":{"selected_model":"model-a","selection_reasoning":"private-canary","annotations":{"note":"private-canary"},"signal_errors":{"signal":"private-canary"}},
		"learning":{"adaptation":{"method":"routing_sampling","reason":"private-canary"}}
	}`)

	redacted := mustRedactChanged(t, body)
	if bytes.Contains(redacted, []byte(private)) {
		t.Fatalf("redacted record still contains private canary: %s", redacted)
	}

	record := decodeRedactedRecord(t, redacted)
	assertRedactedRecordStructure(t, record)
	assertRedactedHallucinationDetail(t, record)
	assertRedactedOutcome(t, record)
}

func mustRedactChanged(t *testing.T, body []byte) []byte {
	t.Helper()
	redacted, changed, err := RedactResponseBody(body)
	if err != nil {
		t.Fatalf("RedactResponseBody() error = %v", err)
	}
	if !changed {
		t.Fatal("RedactResponseBody() changed = false, want true")
	}
	return redacted
}

func decodeRedactedRecord(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var record map[string]any
	if err := json.Unmarshal(body, &record); err != nil {
		t.Fatalf("decode redacted record: %v", err)
	}
	return record
}

func assertRedactedRecordStructure(t *testing.T, record map[string]any) {
	t.Helper()
	if record["decision"] != "safe-decision" {
		t.Fatalf("structural decision changed: %#v", record["decision"])
	}
	if len(record["pii_entities"].([]any)) != 0 || len(record["hallucination_spans"].([]any)) != 0 {
		t.Fatalf("privacy arrays not cleared: pii=%#v hallucination=%#v", record["pii_entities"], record["hallucination_spans"])
	}
}

func assertRedactedHallucinationDetail(t *testing.T, record map[string]any) {
	t.Helper()
	detail := record["hallucination_span_details"].([]any)[0].(map[string]any)
	if detail["text"] != "" || detail["explanation"] != "" || detail["severity"] != float64(2) {
		t.Fatalf("hallucination detail redaction lost structure: %#v", detail)
	}
}

func assertRedactedOutcome(t *testing.T, record map[string]any) {
	t.Helper()
	outcome := record["outcomes"].([]any)[0].(map[string]any)
	if outcome["target_ref"] != "" || outcome["reason"] != "" || len(outcome["metadata"].(map[string]any)) != 0 {
		t.Fatalf("outcome content not cleared: %#v", outcome)
	}
	if outcome["source"] != "user" || outcome["target"] != "model" || outcome["verdict"] != "failed" {
		t.Fatalf("outcome structure changed: %#v", outcome)
	}
}
