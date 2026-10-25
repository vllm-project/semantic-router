package protocolcodec

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func responsesReasoningFrames(t *testing.T) []map[string]any {
	t.Helper()
	body, err := os.ReadFile("testdata/responses-reasoning-parts.sse")
	if err != nil {
		t.Fatal(err)
	}
	var frames []map[string]any
	for _, line := range strings.Split(string(body), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var frame map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame); err != nil {
			t.Fatal(err)
		}
		frames = append(frames, frame)
	}
	return frames
}

func decodeResponsesReasoningFrames(t *testing.T, frames []map[string]any) ([]llmprotocol.Event, error) {
	t.Helper()
	decoder := (OpenAIResponsesCodec{}).NewDecoder(llmprotocol.StreamContext{}, llmprotocol.DefaultPolicy())
	var events []llmprotocol.Event
	for index, frame := range frames {
		frame["sequence_number"] = index
		body, err := encodeSSE(frame["type"].(string), frame)
		if err != nil {
			t.Fatal(err)
		}
		decoded, _, err := decoder.Push(body)
		if err != nil {
			return nil, err
		}
		events = append(events, decoded...)
	}
	_, _, err := decoder.Finalize(nil)
	return events, err
}

func TestResponsesReasoningParts(t *testing.T) {
	for _, mode := range []string{"explicit", "implicit", "implicit with part done", "initial text", "empty text"} {
		t.Run(mode, func(t *testing.T) {
			frames := responsesReasoningFrames(t)
			wantText := "think"
			switch mode {
			case "implicit":
				frames = append(frames[:9], frames[10:]...)
				frames = append(frames[:6], frames[7:]...)
			case "implicit with part done":
				frames = append(frames[:6], frames[7:]...)
			case "initial text":
				frames[6]["part"].(map[string]any)["text"] = "th"
				frames[7]["delta"] = "ink"
			case "empty text":
				wantText = ""
				frames[7]["delta"] = ""
				frames[8]["text"] = ""
				frames[9]["part"].(map[string]any)["text"] = ""
				frames[10]["item"].(map[string]any)["content"].([]any)[0].(map[string]any)["text"] = ""
				frames[17]["response"].(map[string]any)["output"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"] = ""
			}
			events, err := decodeResponsesReasoningFrames(t, frames)
			if err != nil {
				t.Fatal(err)
			}
			text := map[llmprotocol.ReasoningScope]string{}
			indexes := map[llmprotocol.ReasoningScope]int{}
			for _, event := range events {
				if event.Type == llmprotocol.EventReasoningDelta {
					text[event.Content.Reasoning] += event.Delta
					indexes[event.Content.Reasoning] = event.ContentIndex
				}
			}
			if text[llmprotocol.ReasoningScopeText] != wantText || text[llmprotocol.ReasoningScopeSummary] != "summary" ||
				indexes[llmprotocol.ReasoningScopeText] == indexes[llmprotocol.ReasoningScopeSummary] ||
				events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
				t.Fatalf("lost reasoning text, independent indexes or completion: %+v", events)
			}
		})
	}
}

func TestResponsesReasoningPartsRejectInvalidLifecycle(t *testing.T) {
	tests := map[string]func([]map[string]any) []map[string]any{
		"item kind": func(f []map[string]any) []map[string]any {
			f[6]["part"].(map[string]any)["type"] = "output_text"
			return f
		},
		"item id": func(f []map[string]any) []map[string]any { f[6]["item_id"] = "wrong"; return f },
		"index":   func(f []map[string]any) []map[string]any { f[6]["content_index"] = 1; return f },
		"duplicate start": func(f []map[string]any) []map[string]any {
			return append(append([]map[string]any{}, f[:7]...), f[6:]...)
		},
		"duplicate done": func(f []map[string]any) []map[string]any {
			return append(append([]map[string]any{}, f[:10]...), f[9:]...)
		},
		"missing part done":  func(f []map[string]any) []map[string]any { return append(f[:9], f[10:]...) },
		"text done mismatch": func(f []map[string]any) []map[string]any { f[8]["text"] = "wrong"; return f },
		"part done mismatch": func(f []map[string]any) []map[string]any { f[9]["part"].(map[string]any)["text"] = "wrong"; return f },
		"item mismatch": func(f []map[string]any) []map[string]any {
			f[10]["item"].(map[string]any)["content"].([]any)[0].(map[string]any)["text"] = "wrong"
			return f
		},
		"terminal mismatch": func(f []map[string]any) []map[string]any {
			f[17]["response"].(map[string]any)["output"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"] = "wrong"
			return f
		},
	}
	tests["part done before text done"] = func(f []map[string]any) []map[string]any { return append(f[:8], f[9:]...) }
	tests["delta after text done"] = func(f []map[string]any) []map[string]any {
		return append(append([]map[string]any{}, f[:9]...), f[7:]...)
	}
	tests["output index"] = func(f []map[string]any) []map[string]any { f[6]["output_index"] = 1; return f }
	tests["negative index"] = func(f []map[string]any) []map[string]any { f[6]["content_index"] = -1; return f }
	tests["part in message"] = func(f []map[string]any) []map[string]any {
		part := f[12]["part"].(map[string]any)
		part["type"] = "reasoning_text"
		delete(part, "annotations")
		return f
	}
	tests["message final kind"] = func(f []map[string]any) []map[string]any {
		f[16]["item"].(map[string]any)["content"].([]any)[0].(map[string]any)["type"] = "reasoning_text"
		return f
	}
	tests["reasoning final kind"] = func(f []map[string]any) []map[string]any {
		f[10]["item"].(map[string]any)["content"].([]any)[0].(map[string]any)["type"] = "output_text"
		return f
	}
	tests["missing summary done"] = func(f []map[string]any) []map[string]any {
		f = append(f[:9], f[10:]...)
		f = append(f[:6], f[7:]...)
		return append(f[:5], f[6:]...)
	}
	codes := map[string]string{
		"item kind": "stream_item_kind_mismatch", "item id": "stream_item_id_mismatch",
		"index": "stream_content_index_order", "duplicate start": "duplicate_stream_content_start",
		"duplicate done": "duplicate_stream_content_part_done", "missing part done": "stream_item_content_incomplete",
		"text done mismatch": "stream_content_text_mismatch", "part done mismatch": "stream_content_part_mismatch",
		"item mismatch": "stream_item_content_mismatch", "terminal mismatch": "stream_terminal_output_mismatch",
		"part done before text done": "stream_content_text_incomplete", "delta after text done": "stream_content_after_done",
		"output index": "invalid_item_lifecycle", "negative index": "invalid_stream_item_index",
		"part in message": "stream_item_kind_mismatch", "message final kind": "stream_item_content_kind",
		"reasoning final kind": "stream_item_content_kind", "missing summary done": "stream_item_content_incomplete",
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := decodeResponsesReasoningFrames(t, mutate(responsesReasoningFrames(t)))
			assertReasoningPartErrorCode(t, err, codes[name])
		})
	}
}

func TestResponsesReasoningPartsRequireStringText(t *testing.T) {
	for _, index := range []int{6, 9} {
		for _, test := range []struct {
			name  string
			value any
			code  string
		}{
			{name: "absent", code: "stream_required_field"},
			{name: "null", code: "stream_required_field"},
			{name: "number", value: 0, code: "invalid_upstream_json"},
			{name: "boolean", value: false, code: "invalid_upstream_json"},
			{name: "object", value: map[string]any{}, code: "invalid_upstream_json"},
			{name: "array", value: []any{}, code: "invalid_upstream_json"},
		} {
			t.Run(fmt.Sprintf("%s/%d", test.name, index), func(t *testing.T) {
				frames := responsesReasoningFrames(t)
				part := frames[index]["part"].(map[string]any)
				if test.name == "absent" {
					delete(part, "text")
				} else {
					part["text"] = test.value
				}
				_, err := decodeResponsesReasoningFrames(t, frames)
				assertReasoningPartErrorCode(t, err, test.code)
			})
		}
	}
}

func TestResponsesReasoningPartsRequireSequence(t *testing.T) {
	for _, index := range []int{6, 9} {
		for _, test := range []struct {
			name  string
			value any
			code  string
		}{
			{name: "absent", code: "missing_stream_sequence"},
			{name: "null", code: "invalid_stream_sequence"},
			{name: "repeated", value: index - 1, code: "stream_sequence_order"},
		} {
			t.Run(fmt.Sprintf("%s/%d", test.name, index), func(t *testing.T) {
				frames := responsesReasoningFrames(t)[:index+1]
				if test.name == "absent" {
					delete(frames[index], "sequence_number")
				} else {
					frames[index]["sequence_number"] = test.value
				}
				decoder := (OpenAIResponsesCodec{}).NewDecoder(llmprotocol.StreamContext{}, llmprotocol.DefaultPolicy())
				for frameIndex, frame := range frames {
					body, err := encodeSSE(frame["type"].(string), frame)
					if err != nil {
						t.Fatal(err)
					}
					_, _, err = decoder.Push(body)
					if frameIndex < index && err != nil {
						t.Fatal(err)
					}
					if frameIndex == index {
						assertReasoningPartErrorCode(t, err, test.code)
					}
				}
			})
		}
	}
}

func TestResponsesReasoningPartsRejectInvalidType(t *testing.T) {
	for _, index := range []int{6, 9} {
		for _, kind := range []any{nil, "unknown_part", 42, false} {
			t.Run(fmt.Sprintf("%v/%d", kind, index), func(t *testing.T) {
				frames := responsesReasoningFrames(t)
				frames[index]["part"].(map[string]any)["type"] = kind
				if _, err := decodeResponsesReasoningFrames(t, frames); err == nil {
					t.Fatal("invalid content part type was accepted")
				}
			})
		}
		t.Run(fmt.Sprintf("missing/%d", index), func(t *testing.T) {
			frames := responsesReasoningFrames(t)
			delete(frames[index]["part"].(map[string]any), "type")
			if _, err := decodeResponsesReasoningFrames(t, frames); err == nil {
				t.Fatal("missing content part type was accepted")
			}
		})
	}
}

func TestResponsesReasoningPartsRejectForeignVariantFields(t *testing.T) {
	for _, index := range []int{6, 9} {
		for field, value := range map[string]any{
			"refusal":     "foreign variant content",
			"annotations": []any{},
			"file_id":     "file_1",
			"logprobs":    []any{},
		} {
			t.Run(fmt.Sprintf("%s/%d", field, index), func(t *testing.T) {
				frames := responsesReasoningFrames(t)
				frames[index]["part"].(map[string]any)[field] = value
				_, err := decodeResponsesReasoningFrames(t, frames)
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_response_content")
			})
		}
	}
}

func TestResponsesFinalPartNamespaces(t *testing.T) {
	for _, scope := range []responsesContentScope{responsesContentMessage, responsesContentReasoningText, responsesContentReasoningSummary} {
		for _, kind := range []string{"output_text", "refusal", "reasoning_text", "summary_text"} {
			t.Run(string(scope)+"/"+kind, func(t *testing.T) {
				part := responsesContentWire{Type: kind, Text: "text", Refusal: "refusal"}
				gotKind, text, err := responsesFinalPartEvidence(scope, part)
				allowed := scope == responsesContentMessage && (kind == "output_text" || kind == "refusal") ||
					scope == responsesContentReasoningText && kind == "reasoning_text" ||
					scope == responsesContentReasoningSummary && kind == "summary_text"
				if !allowed {
					if err == nil {
						t.Fatal("foreign content kind accepted by namespace")
					}
					return
				}
				wantText := part.Text
				if kind == "refusal" {
					wantText = part.Refusal
				}
				if err != nil || gotKind != kind || text != wantText {
					t.Fatalf("namespace evidence changed: %q %q %v", gotKind, text, err)
				}
			})
		}
	}
}

func assertReasoningPartErrorCode(t *testing.T, err error, code string) {
	t.Helper()
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != code {
		t.Fatalf("got %v, want %s", err, code)
	}
}
