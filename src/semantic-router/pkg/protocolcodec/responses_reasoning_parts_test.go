package protocolcodec

import (
	"encoding/json"
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
		"missing part done":  func(f []map[string]any) []map[string]any { return append(f[:9], f[10:]...) },
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
	tests["part in message"] = func(f []map[string]any) []map[string]any {
		part := f[12]["part"].(map[string]any)
		part["type"] = "reasoning_text"
		delete(part, "annotations")
		return f
	}
	tests["missing summary done"] = func(f []map[string]any) []map[string]any {
		f = append(f[:9], f[10:]...)
		f = append(f[:6], f[7:]...)
		return append(f[:5], f[6:]...)
	}
	codes := map[string]string{
		"item kind": "stream_item_kind_mismatch", "missing part done": "stream_item_content_incomplete",
		"part done mismatch": "stream_content_part_mismatch",
		"item mismatch":      "stream_item_content_mismatch", "terminal mismatch": "stream_terminal_output_mismatch",
		"part in message": "stream_item_kind_mismatch", "missing summary done": "stream_item_content_incomplete",
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := decodeResponsesReasoningFrames(t, mutate(responsesReasoningFrames(t)))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, codes[name])
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
				assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
			})
		}
	}
}

func TestResponsesReasoningPartsRejectInvalidType(t *testing.T) {
	for _, index := range []int{6, 9} {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			frames := responsesReasoningFrames(t)
			frames[index]["part"].(map[string]any)["type"] = "unknown_part"
			_, err := decodeResponsesReasoningFrames(t, frames)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_stream_content_part")
		})
	}
}

func TestResponsesReasoningPartsRejectForeignVariantFields(t *testing.T) {
	for _, index := range []int{6, 9} {
		t.Run(fmt.Sprint(index), func(t *testing.T) {
			frames := responsesReasoningFrames(t)
			frames[index]["part"].(map[string]any)["refusal"] = "foreign variant content"
			_, err := decodeResponsesReasoningFrames(t, frames)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_response_content")
		})
	}
}

func TestResponsesFinalPartNamespaces(t *testing.T) {
	for scope, kind := range map[responsesContentScope]string{
		responsesContentMessage:          "reasoning_text",
		responsesContentReasoningText:    "summary_text",
		responsesContentReasoningSummary: "reasoning_text",
	} {
		t.Run(string(scope), func(t *testing.T) {
			_, _, err := responsesFinalPartEvidence(scope, responsesContentWire{Type: kind, Text: "text"})
			if err == nil {
				t.Fatal("foreign content kind accepted by namespace")
			}
		})
	}
}
