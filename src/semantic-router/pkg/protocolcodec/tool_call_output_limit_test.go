package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// cutChatReply is a buffered Chat reply whose output limit cut its second call
// inside the arguments.
const cutChatReply = `{"id":"chatcmpl-cut","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,` +
	`"message":{"role":"assistant","content":null,"tool_calls":[` +
	`{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}},` +
	`{"id":"call_2","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Lyon\",\"unit\":\"cel"}}]},` +
	`"finish_reason":"length"}],"usage":{"prompt_tokens":10,"completion_tokens":24,"total_tokens":34}}`

// A buffered reply that the output limit cut inside a call reaches every
// client with the limit stop. Chat and Responses clients get the arguments as
// the backend sent them; tool_use.input must be an object, so a Messages client
// gets the members completed before the cut, reported as approximated.
func TestBufferedToolCallCutByTheOutputLimitTranslates(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		target    llmprotocol.WireFormat
		fragments []string
		input     string
	}{
		{llmprotocol.OpenAIChatV1, []string{`"finish_reason":"length"`, `"arguments":"{\"city\":\"Lyon\",\"unit\":\"cel"`}, ""},
		{llmprotocol.OpenAIResponsesV1, []string{`"status":"incomplete"`, `"incomplete_details":{"reason":"max_output_tokens"}`, `"arguments":"{\"city\":\"Lyon\",\"unit\":\"cel"`}, ""},
		{llmprotocol.AnthropicMessagesV1, []string{`"stop_reason":"max_tokens"`}, `{"city":"Lyon"}`},
	}
	for _, test := range tests {
		t.Run(string(test.target), func(t *testing.T) {
			result, err := engine.TranslateResponse(llmprotocol.OpenAIChatV1, test.target, []byte(cutChatReply), func(response *llmprotocol.Response) error {
				response.Model = "routed-model"
				return nil
			})
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			for _, fragment := range test.fragments {
				if !strings.Contains(string(result.Body), fragment) {
					t.Errorf("want %s in:\n%s", fragment, result.Body)
				}
			}
			approximated := slices.ContainsFunc(result.Diagnostics, func(diagnostic llmprotocol.Diagnostic) bool {
				return diagnostic.Field == "content.tool_use.input" && diagnostic.Action == llmprotocol.DiagnosticApproximated
			})
			if approximated != (test.input != "") {
				t.Errorf("approximated tool_use.input diagnostic = %t, want %t: %+v", approximated, test.input != "", result.Diagnostics)
			}
			if test.input == "" {
				return
			}
			var message struct {
				Content []struct {
					ID    string          `json:"id"`
					Input json.RawMessage `json:"input"`
				} `json:"content"`
			}
			if err = json.Unmarshal(result.Body, &message); err != nil {
				t.Fatal(err)
			}
			inputs := map[string]string{}
			for _, block := range message.Content {
				inputs[block.ID] = string(block.Input)
			}
			first := anthropicReplyID(llmprotocol.OpenAIChatV1, anthropicToolUseIDPrefix, "call_1")
			second := anthropicReplyID(llmprotocol.OpenAIChatV1, anthropicToolUseIDPrefix, "call_2")
			if inputs[first] != `{"city":"Paris"}` || inputs[second] != test.input {
				t.Fatalf("tool_use inputs = %v, want call_1 whole and call_2 %s", inputs, test.input)
			}
		})
	}
}

// vLLM's last chunk can carry a call's latest fragment and its tool parser's
// flush of the rest as two entries with one index, and a chunk can carry a
// call's name and its first fragment the same way. They continue one call, in
// order, as clients join them. Entries with one index that name different
// calls still fail.
func TestChatStreamToolEntriesWithOneIndexContinueOneCall(t *testing.T) {
	chunk := func(toolCalls, finish string) string {
		return `data: {"id":"chatcmpl-split","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,` +
			`"delta":{"tool_calls":[` + toolCalls + `]},"logprobs":null,"finish_reason":` + finish + `}]}` + "\n\n"
	}
	named := `{"id":"call_1","type":"function","index":0,"function":{"name":"get_weather"}}`
	fragment := func(arguments string) string {
		encoded, _ := json.Marshal(arguments)
		return `{"index":0,"function":{"arguments":` + string(encoded) + `}}`
	}
	tests := map[string]struct {
		frames []string
		code   string
	}{
		"name and first fragment": {frames: []string{chunk(named+","+fragment(`{"city": "`), "null"), chunk(fragment(`Paris"}`), `"tool_calls"`)}},
		"fragment and flush":      {frames: []string{chunk(named, "null"), chunk(fragment(`{"city": "Par`)+","+fragment(`is"}`), `"length"`)}},
		"the same ID twice": {frames: []string{
			chunk(named+`,{"id":"call_1","index":0,"function":{"arguments":"{\"city\": \"Paris\"}"}}`, `"tool_calls"`),
		}},
		"two IDs": {code: "duplicate_stream_tool_index", frames: []string{
			chunk(named+`,{"id":"call_2","type":"function","index":0,"function":{"name":"get_weather"}}`, "null"),
		}},
		"two names": {code: "duplicate_stream_tool_index", frames: []string{
			chunk(named+`,{"index":0,"function":{"name":"get_time"}}`, "null"),
		}},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background()}, llmprotocol.DefaultPolicy())
			var events []llmprotocol.Event
			var err error
			for _, frame := range append(test.frames, "data: [DONE]\n\n") {
				var decoded []llmprotocol.Event
				decoded, _, err = decoder.Push([]byte(frame))
				events = append(events, decoded...)
				if err != nil {
					break
				}
			}
			if test.code != "" {
				requireProtocolErrorCode(t, err, test.code)
				return
			}
			if err != nil {
				t.Fatalf("push: %v", err)
			}
			var completed *llmprotocol.ToolCall
			for _, event := range events {
				if event.Type == llmprotocol.EventOutputItemCompleted && event.ToolCall != nil {
					completed = event.ToolCall
				}
			}
			if completed == nil || completed.ID != "call_1" || completed.Name != "get_weather" || completed.Arguments != `{"city": "Paris"}` {
				t.Fatalf("completed call = %+v, want call_1 get_weather {\"city\": \"Paris\"}", completed)
			}
		})
	}
}

// toolCallStopReply is a buffered Chat reply that stops for its call after
// completionTokens, as vLLM ends a call its tool parser found even when the
// output limit cut it.
func toolCallStopReply(arguments string, completionTokens int) string {
	encoded, _ := json.Marshal(arguments)
	return fmt.Sprintf(`{"id":"chatcmpl-limit","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,`+
		`"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",`+
		`"function":{"name":"send_email","arguments":%s}}]},"finish_reason":"tool_calls"}],`+
		`"usage":{"prompt_tokens":10,"completion_tokens":%d,"total_tokens":%d}}`, encoded, completionTokens, 10+completionTokens)
}

// toolCallStopStream is the stream of a call the output limit cut inside its
// arguments, ending with finish after completionTokens.
func toolCallStopStream(finish string, completionTokens int) string {
	chunk := func(fields string) string {
		return `data: {"id":"chatcmpl-limit","object":"chat.completion.chunk","created":1,"model":"m",` + fields + "}\n\n"
	}
	return chunk(`"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]`) +
		chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"send_email","arguments":""}}]},"finish_reason":null}]`) +
		chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"body\": \"Dear team, the review moves to ten."}}]},"finish_reason":null}]`) +
		chunk(`"choices":[{"index":0,"delta":{},"finish_reason":"`+finish+`"}]`) +
		chunk(fmt.Sprintf(`"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":%d,"total_tokens":%d}`, completionTokens, 10+completionTokens)) +
		"data: [DONE]\n\n"
}

func engineWithRequestOutputLimit(t *testing.T, limit int64) *Engine {
	t.Helper()
	policy := llmprotocol.DefaultPolicy()
	policy.RequestOutputLimit = limit
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatal(err)
	}
	return engine
}

func translateChatStream(t *testing.T, engine *Engine, target llmprotocol.WireFormat, body string) (string, error) {
	t.Helper()
	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	frames, _, _, err := stream.Push([]byte(body))
	if err == nil {
		var final [][]byte
		final, _, _, err = stream.Finalize(nil)
		frames = append(frames, final...)
	}
	return string(bytes.Join(frames, nil)), err
}

func rerender(*llmprotocol.Response) error { return nil }

// A buffered reply that stops for its call once its output reached the
// request's output limit is a reply the limit cut, as vLLM's is when its tool
// parser closes the cut call as {}. Every client gets exactly what the same
// reply with finish_reason length gets, and the stop is reported as
// approximated.
func TestBufferedToolCallStopAtTheOutputLimitTranslatesAsALengthStop(t *testing.T) {
	limited := engineWithRequestOutputLimit(t, 50)
	arguments := map[string]string{"closed": `{}`, "cut": `{"to": "team@example.com", "body": "Dear team, the review moves to ten.`}
	for name, arguments := range arguments {
		toolCalls := toolCallStopReply(arguments, 50)
		length := strings.Replace(toolCalls, `"finish_reason":"tool_calls"`, `"finish_reason":"length"`, 1)
		for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
			t.Run(name+"/"+string(target), func(t *testing.T) {
				got, err := limited.TranslateResponse(llmprotocol.OpenAIChatV1, target, []byte(toolCalls), rerender)
				if err != nil {
					t.Fatalf("translate: %v", err)
				}
				want, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, target, []byte(length), rerender)
				if err != nil {
					t.Fatalf("translate the length stop: %v", err)
				}
				if string(got.Body) != string(want.Body) {
					t.Fatalf("body =\n%s\nwant the length stop's\n%s", got.Body, want.Body)
				}
				if got.Response.StopReason != llmprotocol.StopMaxTokens || got.Response.SourceStopReason != "tool_calls" {
					t.Fatalf("stop = %s from %q, want max_tokens from tool_calls", got.Response.StopReason, got.Response.SourceStopReason)
				}
				if !slices.ContainsFunc(got.Diagnostics, func(diagnostic llmprotocol.Diagnostic) bool {
					return diagnostic.Field == "response.stop_reason" && diagnostic.Action == llmprotocol.DiagnosticApproximated
				}) {
					t.Fatalf("no approximated response.stop_reason diagnostic: %+v", got.Diagnostics)
				}
			})
		}
	}
	// A Chat client of a Chat backend would otherwise get the backend's bytes,
	// which name the tool-call stop.
	same, err := limited.TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(toolCallStopReply(`{}`, 50)), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(same.Body, []byte(`"finish_reason":"length"`)) || !same.Envelope.ResponseReencodeRequired {
		t.Fatalf("same-format reply = %s, re-encode required %t", same.Body, same.Envelope.ResponseReencodeRequired)
	}
}

// Below the limit, without usage, without a limit, and with several choices,
// whose usage counts them all, the backend's tool-call stop stands, and cut
// arguments under it still fail the reply.
func TestBufferedToolCallStopBelowTheOutputLimitStands(t *testing.T) {
	reply := toolCallStopReply(`{}`, 50)
	for name, test := range map[string]struct {
		limit int64
		reply string
	}{
		"below the limit": {50, toolCallStopReply(`{}`, 49)},
		"no usage":        {50, strings.Replace(reply, `,"usage":{"prompt_tokens":10,"completion_tokens":50,"total_tokens":60}`, "", 1)},
		"no limit":        {0, reply},
		"two choices": {50, strings.Replace(reply, `"finish_reason":"tool_calls"}]`,
			`"finish_reason":"tool_calls"},{"index":1,"message":{"role":"assistant","content":"I will not write."},"finish_reason":"stop"}]`, 1)},
	} {
		t.Run(name, func(t *testing.T) {
			result, err := engineWithRequestOutputLimit(t, test.limit).TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, []byte(test.reply), nil)
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			if result.Response.StopReason != llmprotocol.StopToolCall || string(result.Body) != test.reply {
				t.Fatalf("stop = %s, body =\n%s\nwant the backend's tool-call stop", result.Response.StopReason, result.Body)
			}
		})
	}
	_, err := engineWithRequestOutputLimit(t, 50).TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1,
		[]byte(toolCallStopReply(`{"body": "Dear team`, 49)), nil)
	requireProtocolErrorCode(t, err, "invalid_tool_call")
}

// A stream that ends with tool_calls once its output reached the request's
// limit, as some vLLM tool parsers end a call the limit cut, reaches every
// client as the same stream ending with length. Below the limit its cut
// arguments still end the stream with an error.
func TestChatStreamToolCallStopAtTheOutputLimitTranslatesAsALengthStop(t *testing.T) {
	limited := engineWithRequestOutputLimit(t, 50)
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(target), func(t *testing.T) {
			got, err := translateChatStream(t, limited, target, toolCallStopStream("tool_calls", 50))
			if err != nil {
				t.Fatalf("translate: %v\n%s", err, got)
			}
			want, err := translateChatStream(t, NewBuiltinEngine(), target, toolCallStopStream("length", 50))
			if err != nil {
				t.Fatalf("translate the length stop: %v", err)
			}
			if got != want {
				t.Fatalf("stream =\n%s\nwant the length stop's\n%s", got, want)
			}
			_, err = translateChatStream(t, limited, target, toolCallStopStream("tool_calls", 49))
			requireProtocolErrorCode(t, err, "invalid_stream_tool_arguments")
		})
	}
}

// A Chat client of a Chat backend gets the stream as the backend sent it,
// except that a tool_calls finish once the output reached the request's limit
// reports the limit, as the neutral stream does. The finish waits for the
// usage after it however the stream is split, and only its finish_reason
// changes.
func TestChatPublicStreamFilterReportsAToolCallStopAtTheOutputLimit(t *testing.T) {
	for _, test := range []struct {
		name       string
		limit      int64
		completion int
		finish     string
	}{
		{"at the limit", 50, 50, "length"},
		{"below the limit", 50, 49, "tool_calls"},
		{"no limit", 0, 50, "tool_calls"},
	} {
		for _, includeUsage := range []bool{true, false} {
			for _, split := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/usage=%t/split=%t", test.name, includeUsage, split), func(t *testing.T) {
					upstream := toolCallStopStream("tool_calls", test.completion)
					filter := engineWithRequestOutputLimit(t, test.limit).NewChatPublicStreamFilter(includeUsage)
					pieces := []string{upstream}
					if split {
						pieces = strings.Split(upstream, "")
					}
					var output []byte
					for _, piece := range pieces {
						filtered, err := filter.Push([]byte(piece))
						if err != nil {
							t.Fatal(err)
						}
						output = append(output, filtered...)
					}
					final, err := filter.Finalize()
					if err != nil {
						t.Fatal(err)
					}
					output = append(output, final...)
					want := strings.SplitAfter(upstream, "\n\n")
					if !includeUsage {
						want = slices.Delete(want, 4, 5)
					}
					got := strings.SplitAfter(string(output), "\n\n")
					if len(got) != len(want) {
						t.Fatalf("frames =\n%s\nwant %d frames", output, len(want)-1)
					}
					for index := range want {
						if includeUsage && index < 4 {
							want[index] = strings.Replace(want[index], "}\n\n", `,"usage":null}`+"\n\n", 1)
						}
						if index != 3 {
							if got[index] != want[index] {
								t.Fatalf("frame %d = %q, want %q", index, got[index], want[index])
							}
							continue
						}
						finish, wantFinish := canonicalFrame(t, got[index]), canonicalFrame(t, strings.Replace(want[index], "tool_calls", test.finish, 1))
						if finish != wantFinish {
							t.Fatalf("finish frame = %s, want %s", finish, wantFinish)
						}
					}
				})
			}
		}
	}
}

func canonicalFrame(t *testing.T, frame string) string {
	t.Helper()
	var data any
	if err := json.Unmarshal([]byte(strings.TrimPrefix(strings.TrimSpace(frame), "data: ")), &data); err != nil {
		t.Fatalf("frame %q: %v", frame, err)
	}
	canonical, err := json.Marshal(data)
	if err != nil {
		t.Fatal(err)
	}
	return string(canonical)
}
