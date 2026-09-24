package routerreplay

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestRecorderUTF8Boundaries(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	t.Cleanup(func() { _ = recorder.Close() })
	exerciseRecorderUTF8Boundaries(t, recorder)
}

func exerciseRecorderUTF8Boundaries(t *testing.T, recorder *Recorder) {
	t.Helper()
	recorder.SetCapturePolicy(true, true, 4096)
	recorder.SetMaxToolTraceBytes(4096)
	for _, test := range []struct {
		name, prefix, suffix string
	}{
		{"Chinese", strings.Repeat("a", 4095), "中文"},
		{"emoji", strings.Repeat("b", 4094), "🙂done"},
	} {
		t.Run(test.name, func(t *testing.T) {
			input := test.prefix + test.suffix
			trace := &ToolTrace{Steps: []ToolTraceStep{{Arguments: input, Output: input}}}
			id, err := recorder.AddRecord(RoutingRecord{
				RequestBody: input, ResponseBody: input,
				Prompt: input, ToolDefinitions: input, ToolTrace: trace,
			})
			if err != nil {
				t.Fatal(err)
			}
			record, found := recorder.GetRecord(id)
			if !found {
				t.Fatal("captured record is missing")
			}
			assertUTF8Bodies(t, record, test.prefix)
			assertUTF8Text(t, record.Prompt, test.prefix, record.PromptTruncated)
			assertUTF8Text(t, record.ToolDefinitions, test.prefix, record.ToolDefinitionsTruncated)
			assertUTF8Trace(t, record.ToolTrace, test.prefix)

			attachedID, addErr := recorder.AddRecord(RoutingRecord{RequestBody: "before", ResponseBody: "before"})
			if addErr != nil {
				t.Fatal(addErr)
			}
			if attachErr := recorder.AttachRequest(attachedID, []byte(input)); attachErr != nil {
				t.Fatal(attachErr)
			}
			if attachErr := recorder.AttachResponse(attachedID, []byte(input)); attachErr != nil {
				t.Fatal(attachErr)
			}
			if updateErr := recorder.UpdateToolTrace(attachedID, ToolTrace{Steps: []ToolTraceStep{{Arguments: input, Output: input}}}); updateErr != nil {
				t.Fatal(updateErr)
			}
			updated, found := recorder.GetRecord(attachedID)
			if !found {
				t.Fatal("updated record is missing")
			}
			assertUTF8Bodies(t, updated, test.prefix)
			assertUTF8Trace(t, updated.ToolTrace, test.prefix)
		})
	}
}

func assertUTF8Bodies(t *testing.T, record RoutingRecord, want string) {
	t.Helper()
	assertUTF8Text(t, record.RequestBody, want, record.RequestBodyTruncated)
	assertUTF8Text(t, record.ResponseBody, want, record.ResponseBodyTruncated)
}

func assertUTF8Trace(t *testing.T, trace *ToolTrace, want string) {
	t.Helper()
	if trace == nil || len(trace.Steps) != 1 {
		t.Fatal("tool trace is missing")
	}
	assertUTF8Text(t, trace.Steps[0].Arguments, want, trace.Steps[0].Truncated)
	assertUTF8Text(t, trace.Steps[0].Output, want, trace.Steps[0].Truncated)
}

func assertUTF8Text(t *testing.T, got, want string, truncated bool) {
	t.Helper()
	if got != want || !utf8.ValidString(got) || len(got) > 4096 || !truncated {
		t.Fatalf("invalid captured prefix: bytes=%d want=%d valid=%v truncated=%v", len(got), len(want), utf8.ValidString(got), truncated)
	}
}

func TestReplayTextNormalizationAndExactBoundaries(t *testing.T) {
	for _, test := range []struct {
		input, want string
		limit       int
		truncated   bool
	}{
		{"中文🙂", "中文🙂", 10, false},
		{"中文🙂", "中文", 9, true},
		{"中", "", 1, true},
		{"a\xffb", "a\uFFFDb", 0, false},
		{"a\xffb", "a\uFFFDb", 5, false},
		{"a\xffb", "a\uFFFD", 4, true},
	} {
		got, truncated := truncateString(test.input, test.limit)
		if got != test.want || truncated != test.truncated || !utf8.ValidString(got) {
			t.Fatalf("input=%q limit=%d got=%q truncated=%v", test.input, test.limit, got, truncated)
		}
	}
	got, truncated := applyBodyCapturePolicy("中文", true, true, 4096)
	if got != "中文" || !truncated {
		t.Fatal("an existing body truncation flag was lost")
	}
	record := RoutingRecord{Prompt: "a\xffb", ToolDefinitions: "a\xffb", PromptTruncated: true, ToolDefinitionsTruncated: true}
	applyMaxToolTraceBytes(&record, 0)
	if record.Prompt != "a\uFFFDb" || record.ToolDefinitions != "a\uFFFDb" {
		t.Fatal("unbounded structured text must still be valid UTF-8")
	}
	if !record.PromptTruncated || !record.ToolDefinitionsTruncated {
		t.Fatal("existing structured text truncation flags were lost")
	}
}
