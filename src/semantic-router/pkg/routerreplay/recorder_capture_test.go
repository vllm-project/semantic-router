package routerreplay

import (
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
