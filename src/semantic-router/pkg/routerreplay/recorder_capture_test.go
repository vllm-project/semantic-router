package routerreplay

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestRecorderAddRecordDropsBodiesWhenCaptureDisabled(t *testing.T) {
	// Regression test for #2748: with capture disabled, AddRecord used to pass
	// bodies through to storage untouched (it only skipped truncation), so
	// capture_request_body: false still persisted the full request body.
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)

	id, err := recorder.AddRecord(RoutingRecord{
		RequestID:             "req-capture-off",
		RequestBody:           `{"messages":[{"role":"user","content":"my ssn is 123-45-6789"}]}`,
		RequestBodyTruncated:  true,
		ResponseBody:          `{"choices":[{"message":{"content":"generated text"}}]}`,
		ResponseBodyTruncated: true,
	})
	if err != nil {
		t.Fatalf("AddRecord: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}
	if rec.RequestBody != "" {
		t.Errorf("expected empty RequestBody with capture disabled, got %q", rec.RequestBody)
	}
	if rec.RequestBodyTruncated {
		t.Error("expected RequestBodyTruncated=false when no body is stored")
	}
	if rec.ResponseBody != "" {
		t.Errorf("expected empty ResponseBody with capture disabled, got %q", rec.ResponseBody)
	}
	if rec.ResponseBodyTruncated {
		t.Error("expected ResponseBodyTruncated=false when no body is stored")
	}
}

func TestRecorderAddRecordDropsBodiesWithoutCapturePolicy(t *testing.T) {
	// A recorder that was never given a capture policy defaults to
	// capture-off and must not store bodies either, matching the
	// AttachRequest/AttachResponse default.
	recorder := NewRecorder(store.NewMemoryStore(10, 0))

	id, err := recorder.AddRecord(RoutingRecord{
		RequestID:    "req-no-policy",
		RequestBody:  `{"messages":[{"role":"user","content":"hello"}]}`,
		ResponseBody: `{"choices":[]}`,
	})
	if err != nil {
		t.Fatalf("AddRecord: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}
	if rec.RequestBody != "" || rec.ResponseBody != "" {
		t.Errorf("expected no stored bodies without a capture policy, got request=%q response=%q",
			rec.RequestBody, rec.ResponseBody)
	}
}

func TestRecorderAddRecordKeepsAndTruncatesBodiesWhenCaptureEnabled(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(true, true, 10)

	id, err := recorder.AddRecord(RoutingRecord{
		RequestID:    "req-capture-on",
		RequestBody:  "0123456789abcdef",
		ResponseBody: "fedcba9876543210",
	})
	if err != nil {
		t.Fatalf("AddRecord: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}
	if rec.RequestBody != "0123456789" {
		t.Errorf("expected RequestBody truncated to 10 bytes, got %q", rec.RequestBody)
	}
	if !rec.RequestBodyTruncated {
		t.Error("expected RequestBodyTruncated=true")
	}
	if rec.ResponseBody != "fedcba9876" {
		t.Errorf("expected ResponseBody truncated to 10 bytes, got %q", rec.ResponseBody)
	}
	if !rec.ResponseBodyTruncated {
		t.Error("expected ResponseBodyTruncated=true")
	}
}

// Every write path keeps the text of an uncaptured side out of the record:
// the prompt, tool definitions, trace steps and detector excerpts. A step
// whose source names neither side is stored only when both are captured.
func TestRecorderCaptureSwitchesCoverEveryTextField(t *testing.T) {
	steps := func() []ToolTraceStep {
		return []ToolTraceStep{
			{Type: "user_input", Source: ToolTraceSourceRequest, Text: "request-text"},
			{Type: "assistant_tool_call", Source: ToolTraceSourceStream, ToolName: "lookup", ToolCallID: "call-1", Arguments: "response-args", RawArguments: "response-args"},
			{Type: "client_tool_result", Source: "", ToolCallID: "call-1", Status: "succeeded", Output: "unknown-output"},
		}
	}
	for _, capture := range []struct{ request, response bool }{{false, false}, {true, false}, {false, true}, {true, true}} {
		recorder := NewRecorder(store.NewMemoryStore(10, 0))
		recorder.SetCapturePolicy(capture.request, capture.response, 4096)
		id, err := recorder.AddRecord(RoutingRecord{Prompt: "prompt", ToolDefinitions: "definitions", ToolTrace: &ToolTrace{Steps: steps()}})
		if err != nil {
			t.Fatal(err)
		}
		if err := recorder.AttachPrompt(id, "attached-prompt"); err != nil {
			t.Fatal(err)
		}
		if err := recorder.UpdateHallucinationStatus(id, true, 0.9, []string{"excerpt"},
			[]HallucinationSpan{{Text: "excerpt", Explanation: "why", Start: 1, End: 8, Severity: 2}}); err != nil {
			t.Fatal(err)
		}
		added, _ := recorder.GetRecord(id)
		if err := recorder.UpdateToolTrace(id, ToolTrace{Steps: steps()}); err != nil {
			t.Fatal(err)
		}
		updated, _ := recorder.GetRecord(id)

		both := capture.request && capture.response
		for _, check := range []struct {
			field string
			got   bool
			want  bool
		}{
			{"prompt", updated.Prompt != "", capture.request},
			{"tool definitions", added.ToolDefinitions != "", capture.request},
			{"hallucination spans", len(updated.HallucinationSpans) > 0, capture.response},
			{"span text", updated.HallucinationSpanDetails[0].Text+updated.HallucinationSpanDetails[0].Explanation != "", capture.response},
			{"added request step", added.ToolTrace.Steps[0].Text != "", capture.request},
			{"added stream step", added.ToolTrace.Steps[1].Arguments+added.ToolTrace.Steps[1].RawArguments != "", capture.response},
			{"added unknown step", added.ToolTrace.Steps[2].Output != "", both},
			{"updated request step", updated.ToolTrace.Steps[0].Text != "", capture.request},
			{"updated stream step", updated.ToolTrace.Steps[1].Arguments != "", capture.response},
			{"updated unknown step", updated.ToolTrace.Steps[2].Output != "", both},
		} {
			if check.got != check.want {
				t.Errorf("request=%t response=%t: %s stored=%t, want %t", capture.request, capture.response, check.field, check.got, check.want)
			}
		}
		span := updated.HallucinationSpanDetails[0]
		step := updated.ToolTrace.Steps[1]
		if span.Start != 1 || span.End != 8 || span.Severity != 2 || step.ToolName != "lookup" || step.ToolCallID != "call-1" ||
			updated.ToolTrace.Steps[2].Status != "succeeded" || step.ContentRedacted {
			t.Errorf("request=%t response=%t: metadata lost, or left-out text flagged as redacted: %+v %+v", capture.request, capture.response, span, step)
		}
	}
}

// A fallback execution record keeps provider replies only when the response
// is captured; one that cannot be read is not stored as it came.
func TestRecorderCaptureOffKeepsProviderRepliesOutOfOutcomes(t *testing.T) {
	execution := `{"request_id":"r","attempts":[{"model":"m","status_code":503,"error":"upstream returned status 503: reply-text"},{"model":"n","status_code":0,"error":""}]}`
	for _, test := range []struct {
		name, record, want string
		capture            bool
	}{
		{"captured", execution, "reply-text", true},
		{"not captured", execution, `"error":"upstream returned status 503"`, false},
		{"unreadable", `{"attempts":"reply-text"`, "", false},
	} {
		recorder := NewRecorder(store.NewMemoryStore(10, 0))
		recorder.SetCapturePolicy(test.capture, test.capture, 4096)
		id, err := recorder.AddRecord(RoutingRecord{})
		if err != nil {
			t.Fatal(err)
		}
		metadata := map[string]string{fallbackExecutionRecordKey: test.record, "final_status": "failed"}
		if err := recorder.AppendOutcome(id, Outcome{Source: "fallback", Metadata: metadata}); err != nil {
			t.Fatal(err)
		}
		record, _ := recorder.GetRecord(id)
		stored := record.Outcomes[0].Metadata
		if got := stored[fallbackExecutionRecordKey]; !strings.Contains(got, test.want) || (!test.capture && strings.Contains(got, "reply-text")) {
			t.Errorf("%s: execution record %q, want it to contain %q", test.name, got, test.want)
		}
		if stored["final_status"] != "failed" || metadata[fallbackExecutionRecordKey] != test.record {
			t.Errorf("%s: other metadata lost or the caller's map changed: %v", test.name, stored)
		}
	}
}
