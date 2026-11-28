package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func emptyArgumentsChatStream(arguments []string, finish string, parallel bool, tokens int) []string {
	chunk := func(delta, reason string) string {
		return fmt.Sprintf("data: {\"id\":\"chatcmpl-empty\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":%s,\"finish_reason\":%s}]}\n\n", delta, reason)
	}
	calls := `{"index":0,"id":"call_empty","type":"function","function":{"name":"task_status"}}`
	if parallel {
		calls += `,{"index":1,"id":"call_other","type":"function","function":{"name":"probe"}}`
	}
	frames := []string{chunk(`{"tool_calls":[`+calls+`]}`, "null")}
	for _, argument := range arguments {
		encoded, _ := json.Marshal(argument)
		frames = append(frames, chunk(`{"tool_calls":[{"index":0,"function":{"arguments":`+string(encoded)+`}}]}`, "null"))
	}
	if parallel {
		for _, argument := range []string{`{"q":`, `"ok"}`} {
			encoded, _ := json.Marshal(argument)
			frames = append(frames, chunk(`{"tool_calls":[{"index":1,"function":{"arguments":`+string(encoded)+`}}]}`, "null"))
		}
	}
	encodedFinish, _ := json.Marshal(finish)
	frames = append(frames, chunk(`{}`, string(encodedFinish)), fmt.Sprintf("data: {\"id\":\"chatcmpl-empty\",\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n", tokens, tokens+4), "data: [DONE]\n\n")
	return frames
}

func translateEmptyArgumentsChunks(t *testing.T, engine *Engine, chunks []string, split bool) ([][]byte, error) {
	t.Helper()
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	if split {
		chunks = splitGoldenChunksByByte(chunks)
	}
	var frames [][]byte
	for _, chunk := range chunks {
		output, _, _, pushErr := stream.Push([]byte(chunk))
		frames = append(frames, output...)
		if pushErr != nil {
			return frames, pushErr
		}
	}
	final, _, _, err := stream.Finalize(nil)
	return append(frames, final...), err
}

func TestChatResponsesFinishedEmptyArguments(t *testing.T) {
	tests := map[string]struct {
		arguments []string
		want      string
	}{
		"omitted":                  {nil, "{}"},
		"empty":                    {[]string{""}, "{}"},
		"whitespace":               {[]string{" ", "\t\r\n"}, "{}"},
		"explicit object":          {[]string{"{}"}, "{}"},
		"fragmented object":        {[]string{`{"q":`, `"ok"}`}, `{"q":"ok"}`},
		"whitespace before object": {[]string{" \n", `{"q":`, `"ok"}`}, " \n{\"q\":\"ok\"}"},
	}
	for name, test := range tests {
		for _, split := range []bool{false, true} {
			for _, parallel := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/split=%t/parallel=%t", name, split, parallel), func(t *testing.T) {
					frames, err := translateEmptyArgumentsChunks(t, NewBuiltinEngine(), emptyArgumentsChatStream(test.arguments, "tool_calls", parallel, 3), split)
					if err != nil {
						t.Fatal(err)
					}
					want := map[string]string{"call_empty": test.want}
					if parallel {
						want["call_other"] = `{"q":"ok"}`
					}
					assertEmptyArgumentsResponses(t, frames, want)
				})
			}
		}
	}
}

func assertEmptyArgumentsResponses(t *testing.T, frames [][]byte, want map[string]string) {
	t.Helper()
	ids := map[string]string{}
	deltas := map[string]string{}
	deltaCounts := map[string]int{}
	indexes := map[string]int{}
	done := map[string]string{}
	items := map[string]string{}
	terminal := false
	for _, frame := range frames {
		parsed, err := parseSSEFrameAtPosition(frame, 1<<20, false)
		if err != nil {
			t.Fatal(err)
		}
		var wire responsesEventWire
		if err := json.Unmarshal(parsed.Data, &wire); err != nil {
			t.Fatal(err)
		}
		switch wire.Type {
		case "response.output_item.added", "response.output_item.done":
			var item responsesItemWire
			if err := json.Unmarshal(wire.Item, &item); err != nil {
				t.Fatal(err)
			}
			ids[item.ID] = item.CallID
			if wire.OutputIndex == nil {
				t.Fatal("missing output index")
			}
			if index, exists := indexes[item.ID]; exists && index != *wire.OutputIndex {
				t.Fatal("output index changed")
			}
			indexes[item.ID] = *wire.OutputIndex
			if item.Name != map[string]string{"call_empty": "task_status", "call_other": "probe"}[item.CallID] {
				t.Fatalf("tool name changed: %+v", item)
			}
			if wire.Type == "response.output_item.done" {
				items[item.CallID] = item.Arguments
			}
		case "response.function_call_arguments.delta":
			if strings.TrimSpace(wire.Delta) == "" {
				t.Fatalf("leaked empty argument delta: %q", wire.Delta)
			}
			if wire.OutputIndex == nil || *wire.OutputIndex != indexes[wire.ItemID] {
				t.Fatal("argument delta output index changed")
			}
			deltas[wire.ItemID] += wire.Delta
			deltaCounts[wire.ItemID]++
		case "response.function_call_arguments.done":
			if wire.OutputIndex == nil || *wire.OutputIndex != indexes[wire.ItemID] {
				t.Fatal("arguments done output index changed")
			}
			done[wire.ItemID] = wire.Arguments
		case "response.completed":
			terminal = true
			if wire.Response.Status != "completed" {
				t.Fatalf("status = %s", wire.Response.Status)
			}
			var output []json.RawMessage
			if err := json.Unmarshal(wire.Response.Output, &output); err != nil {
				t.Fatal(err)
			}
			for _, raw := range output {
				var item responsesItemWire
				if err := json.Unmarshal(raw, &item); err != nil {
					t.Fatal(err)
				}
				if item.Arguments != want[item.CallID] {
					t.Fatalf("terminal call = %+v", item)
				}
			}
			if len(output) != len(want) {
				t.Fatalf("terminal output count = %d", len(output))
			}
		case "response.incomplete", "response.failed", "error":
			t.Fatalf("unexpected terminal event: %s", frame)
		}
	}
	if !terminal || len(items) != len(want) || len(done) != len(want) {
		t.Fatalf("missing completion: terminal=%t items=%v done=%v", terminal, items, done)
	}
	for id, callID := range ids {
		if want[callID] == "{}" && deltaCounts[id] != 1 {
			t.Fatalf("normalized call %s emitted %d deltas", callID, deltaCounts[id])
		}
		if callID == "call_empty" && indexes[id] != 0 || callID == "call_other" && indexes[id] != 1 {
			t.Fatalf("parallel call index changed: %v", indexes)
		}
		if deltas[id] != want[callID] || done[id] != want[callID] || items[callID] != want[callID] {
			t.Fatalf("call %s: deltas=%q done=%q item=%q want=%q", callID, deltas[id], done[id], items[callID], want[callID])
		}
	}
}

func TestChatResponsesNullArgumentPlaceholder(t *testing.T) {
	for _, test := range []struct {
		name      string
		arguments []string
		want      string
	}{
		{"object follows", []string{`{"q":`, `"ok"}`}, `{"q":"ok"}`},
		{"empty object follows", []string{"{}"}, "{}"},
		{"nothing follows", nil, ""},
		{"empty follows", []string{""}, ""},
		{"whitespace follows", []string{" \t"}, ""},
		{"array follows", []string{"[]"}, ""},
		{"malformed follows", []string{"{bad}"}, ""},
	} {
		for _, split := range []bool{false, true} {
			for _, parallel := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/split=%t/parallel=%t", test.name, split, parallel), func(t *testing.T) {
					chunks := emptyArgumentsChatStream(test.arguments, "tool_calls", parallel, 3)
					chunks[0] = strings.Replace(chunks[0], `"name":"task_status"`, `"name":"task_status","arguments":null`, 1)
					// Arrival of null alone must not abort an unfinished call.
					stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{Context: t.Context()})
					if err != nil {
						t.Fatal(err)
					}
					if _, _, _, err = stream.Push([]byte(chunks[0])); err != nil {
						t.Fatal(err)
					}
					frames, err := translateEmptyArgumentsChunks(t, NewBuiltinEngine(), chunks, split)
					if test.want == "" {
						requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
						return
					}
					if err != nil {
						t.Fatal(err)
					}
					want := map[string]string{"call_empty": test.want}
					if parallel {
						want["call_other"] = `{"q":"ok"}`
					}
					assertEmptyArgumentsResponses(t, frames, want)
				})
			}
		}
	}
}

func TestChatResponsesDeferredToolCompletionUpstreamError(t *testing.T) {
	for _, split := range []bool{false, true} {
		for _, flushAtEOF := range []bool{false, true} {
			t.Run(fmt.Sprintf("split=%t/flushAtEOF=%t", split, flushAtEOF), func(t *testing.T) {
				stream, err := engineWithRequestOutputLimit(t, 3).NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.StreamContext{Context: t.Context()})
				if err != nil {
					t.Fatal(err)
				}
				decoder := stream.decoder.(*chatStreamDecoder)
				chunks := emptyArgumentsChatStream(nil, "tool_calls", false, 2)
				chunks = chunks[:len(chunks)-2] // finish_reason, but no usage or [DONE]
				chunks = append(chunks, "data: {\"error\":{\"type\":\"server_error\",\"code\":\"provider_failure\",\"message\":\"original upstream failure\"}}\n\n")
				if flushAtEOF {
					chunks[len(chunks)-1] = strings.TrimRight(chunks[len(chunks)-1], "\n")
				}
				if split {
					chunks = splitGoldenChunksByByte(chunks)
				}
				var frames [][]byte
				var events []llmprotocol.Event
				for _, chunk := range chunks {
					output, decoded, _, pushErr := stream.Push([]byte(chunk))
					if pushErr != nil {
						t.Fatal(pushErr)
					}
					frames = append(frames, output...)
					events = append(events, decoded...)
				}
				if !decoder.deferredToolCompletion {
					t.Fatal("test did not exercise deferred tool completion")
				}
				final, decoded, _, finalErr := stream.Finalize(nil)
				if finalErr != nil {
					t.Fatalf("EOF after upstream error: %v", finalErr)
				}
				frames = append(frames, final...)
				events = append(events, decoded...)
				failures := 0
				for _, event := range events {
					if event.Type == llmprotocol.EventResponseFailed {
						failures++
						if event.Error == nil || event.Error.Code != "provider_failure" || event.Error.Message != "original upstream failure" {
							t.Fatalf("original error changed: %+v", event.Error)
						}
					}
				}
				body := string(bytes.Join(frames, nil))
				if failures != 1 || strings.Count(body, `"type":"error"`) != 1 || strings.Contains(body, `"type":"response.completed"`) || strings.Contains(body, "stream_terminal") {
					t.Fatalf("upstream error not handled exactly once: failures=%d body=%s", failures, body)
				}
				if remaining, _, finalizeErr := decoder.Finalize(nil); finalizeErr != nil || len(remaining) != 0 {
					t.Fatalf("terminal decoder did not finalize cleanly: events=%v err=%v", remaining, finalizeErr)
				}
			})
		}
	}
}

func TestChatResponsesEmptyArgumentsBoundaries(t *testing.T) {
	for name, arguments := range map[string][]string{"null": {"null"}, "array": {"[]"}, "scalar": {"1"}, "malformed": {"{bad}"}, "cut object": {`{"q":`}} {
		t.Run(name, func(t *testing.T) {
			_, err := translateEmptyArgumentsChunks(t, NewBuiltinEngine(), emptyArgumentsChatStream(arguments, "tool_calls", false, 3), true)
			requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
		})
	}
	for _, arguments := range [][]string{nil, {" \t"}, {`{"q":`}} {
		for _, split := range []bool{false, true} {
			for _, finish := range []string{"length", "tool_calls"} {
				t.Run(fmt.Sprintf("limit/%q/%s/split=%t", arguments, finish, split), func(t *testing.T) {
					frames, err := translateEmptyArgumentsChunks(t, engineWithRequestOutputLimit(t, 3), emptyArgumentsChatStream(arguments, finish, false, 3), split)
					if err != nil {
						t.Fatal(err)
					}
					body := string(bytes.Join(frames, nil))
					if !strings.Contains(body, `"type":"response.incomplete"`) || !strings.Contains(body, `"reason":"max_output_tokens"`) || strings.Contains(body, `"type":"response.completed"`) {
						t.Fatalf("not an output-limit cut: %s", body)
					}
				})
			}
		}
	}
	for _, tokens := range []int{2, 3} {
		t.Run(fmt.Sprintf("normal stop with trailing usage/%d", tokens), func(t *testing.T) {
			limit := int64(4)
			frames, err := translateEmptyArgumentsChunks(t, engineWithRequestOutputLimit(t, limit), emptyArgumentsChatStream([]string{" \t"}, "tool_calls", true, tokens), true)
			if err != nil {
				t.Fatal(err)
			}
			assertEmptyArgumentsResponses(t, frames, map[string]string{"call_empty": "{}", "call_other": `{"q":"ok"}`})
		})
	}
	for _, finish := range []string{"length", "tool_calls"} {
		t.Run("wire null at output limit/"+finish, func(t *testing.T) {
			chunks := emptyArgumentsChatStream(nil, finish, false, 3)
			chunks[0] = strings.Replace(chunks[0], `"name":"task_status"`, `"name":"task_status","arguments":null`, 1)
			frames, err := translateEmptyArgumentsChunks(t, engineWithRequestOutputLimit(t, 3), chunks, true)
			if err != nil {
				t.Fatal(err)
			}
			body := string(bytes.Join(frames, nil))
			if !strings.Contains(body, `"type":"response.incomplete"`) || !strings.Contains(body, `"reason":"max_output_tokens"`) || strings.Contains(body, `"type":"response.completed"`) {
				t.Fatalf("null placeholder was not preserved at limit: %s", body)
			}
		})
	}
	for _, finish := range []string{"stop", "tool_calls"} {
		t.Run("normal finish/"+finish, func(t *testing.T) {
			chunks := emptyArgumentsChatStream(nil, finish, false, 3)
			// No usage is also a normal completion: no inferred limit evidence.
			chunks = append(chunks[:len(chunks)-2], chunks[len(chunks)-1])
			frames, err := translateEmptyArgumentsChunks(t, engineWithRequestOutputLimit(t, 3), chunks, true)
			if err != nil {
				t.Fatal(err)
			}
			assertEmptyArgumentsResponses(t, frames, map[string]string{"call_empty": "{}"})
		})
	}
	t.Run("missing terminal sentinel", func(t *testing.T) {
		chunks := emptyArgumentsChatStream(nil, "tool_calls", false, 2)
		frames, err := translateEmptyArgumentsChunks(t, engineWithRequestOutputLimit(t, 3), chunks[:len(chunks)-1], true)
		if err != nil {
			t.Fatal(err)
		}
		body := string(bytes.Join(frames, nil))
		if !strings.Contains(body, `"code":"stream_incomplete"`) || strings.Contains(body, `"type":"response.completed"`) {
			t.Fatalf("unfinished stream normalized successfully: %s", body)
		}
	})
	// Other translation directions retain their existing rejection contract.
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		_, err := translateChatStream(t, NewBuiltinEngine(), target, strings.Join(emptyArgumentsChatStream(nil, "tool_calls", false, 3), ""))
		requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
	}
}
