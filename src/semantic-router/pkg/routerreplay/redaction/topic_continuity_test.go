package redaction

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestRedactionKeepsTopicContinuityReceipts(t *testing.T) {
	receipt := map[string]any{
		"signal": "topic_boundary", "class": "change", "reason": "change_explicit_marker",
		"coverage": "full", "confidence": 0.9, "classify_us": float64(4),
	}
	body, err := json.Marshal(map[string]any{
		"request_body":      "secret prompt",
		"route_diagnostics": map[string]any{"topic_continuity": []any{receipt}},
	})
	if err != nil {
		t.Fatal(err)
	}
	redacted, _, err := RedactResponseBody(body)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(redacted, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded["request_body"] == "secret prompt" {
		t.Fatal("precondition: redaction should clear the request body")
	}
	diagnostics := decoded["route_diagnostics"].(map[string]any)
	got := diagnostics["topic_continuity"].([]any)[0]
	if !reflect.DeepEqual(got, receipt) {
		t.Fatalf("redaction changed the content-free receipt: %v", got)
	}
}
