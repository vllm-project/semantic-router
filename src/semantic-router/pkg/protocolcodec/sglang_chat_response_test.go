package protocolcodec

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// SGLang 0.5.22 / Qwen3-0.6B, boundary-r2 sglang-A chat_nonstream and
// chat_stream recordings. Stream bytes are unchanged, with one SSE event per
// recorded network chunk. Buffered bytes omit only flat usage.reasoning_tokens
// (covered separately by #4914); matched_stop and metadata remain as recorded.
func sglangChatCapture(t *testing.T, name string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "providers", "sglang-0.5.22-chat-"+name))
	if err != nil {
		t.Fatalf("%v (cause: %v)", err, errors.Unwrap(err))
	}
	return body
}

const sglangHaiku = "Golden leaves fall,  \nThe air turns crisp,  \nA season of change."

func TestSGLangRecordedBufferedChatResponse(t *testing.T) {
	response, _, diagnostics, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, sglangChatCapture(t, "out.json"))
	if err != nil {
		t.Fatalf("%v (cause: %v)", err, errors.Unwrap(err))
	}
	_, text := collectDecodedText(response.Output)
	if text != sglangHaiku || response.StopReason != llmprotocol.StopEndTurn || response.MatchedStopSequence != "" ||
		response.Usage.Total.Value == nil || *response.Usage.Total.Value != 36 {
		t.Fatalf("unexpected decoded response: %+v", response)
	}
	assertDiagnosticFields(t, diagnostics, "metadata")
}

func TestSGLangRecordedChatStreamChunks(t *testing.T) {
	decoder := OpenAIChatCodec{}.NewDecoder(llmprotocol.StreamContext{}, llmprotocol.DefaultPolicy())
	accumulator := newResponseAccumulator()
	completed := 0
	for _, chunk := range bytes.SplitAfter(sglangChatCapture(t, "stream.sse"), []byte("\n\n")) {
		events, _, err := decoder.Push(chunk)
		if err != nil {
			t.Fatalf("%v (cause: %v)", err, errors.Unwrap(err))
		}
		for _, event := range events {
			if event.Type == llmprotocol.EventResponseCompleted {
				completed++
			}
		}
		if err := accumulator.apply(events); err != nil {
			t.Fatal(err)
		}
	}
	events, _, err := decoder.Finalize(nil)
	if err != nil || len(events) != 0 {
		t.Fatalf("finalize: events=%+v err=%v", events, err)
	}
	response, err := accumulator.response()
	_, text := collectDecodedText(response.Output)
	if err != nil || text != sglangHaiku || completed != 1 || response.StopReason != llmprotocol.StopEndTurn || response.MatchedStopSequence != "" {
		t.Fatalf("stream: response=%+v completions=%d err=%v", response, completed, err)
	}
}

func TestSGLangChatMatchedStopShapes(t *testing.T) {
	for _, stop := range []string{"null", "0", "151645", `"STOP"`} {
		t.Run(stop, func(t *testing.T) {
			buffered := bytes.ReplaceAll(sglangChatCapture(t, "out.json"), []byte("151645"), []byte(stop))
			response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, buffered)
			if err != nil || response.StopReason != llmprotocol.StopEndTurn || response.MatchedStopSequence != "" {
				t.Fatalf("buffered: response=%+v err=%v", response, err)
			}
			stream := bytes.ReplaceAll(sglangChatCapture(t, "stream.sse"), []byte("151645"), []byte(stop))
			response, _, err = NewBuiltinEngine().DecodeResponseStream(llmprotocol.OpenAIChatV1, stream, llmprotocol.StreamContext{})
			if err != nil || response.StopReason != llmprotocol.StopEndTurn || response.MatchedStopSequence != "" {
				t.Fatalf("stream: response=%+v err=%v", response, err)
			}
		})
	}
}

func TestSGLangChatWeightVersions(t *testing.T) {
	// Isolate the metadata failure independently of matched_stop.
	body := bytes.ReplaceAll(sglangChatCapture(t, "out.json"), []byte(`,"matched_stop":151645`), nil)
	versions := `,"weight_versions":[{"version":"default","start":0,"end":17}]`
	for name, replacement := range map[string]string{"recorded": versions, "absent": "", "empty": `,"weight_versions":[]`} {
		t.Run(name, func(t *testing.T) {
			response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, bytes.ReplaceAll(body, []byte(versions), []byte(replacement)))
			_, text := collectDecodedText(response.Output)
			if err != nil || text != sglangHaiku {
				t.Fatalf("text=%q err=%v (cause: %v)", text, err, errors.Unwrap(err))
			}
		})
	}
}

func TestSGLangChatResponseCompatibilityControls(t *testing.T) {
	for name, metadata := range map[string]string{"absent": "", "null": `,"metadata":null`, "empty": `,"metadata":{}`, "string": `,"metadata":{"weight_version":"default","weight_versions":"legacy"}`} {
		t.Run(name, func(t *testing.T) {
			body := []byte(`{"id":"r","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}]` + metadata + `}`)
			response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, body)
			if err != nil || response.StopReason != llmprotocol.StopEndTurn {
				t.Fatalf("response=%+v err=%v", response, err)
			}
		})
	}
	for name, extra := range map[string]string{
		"unknown": `"unrelated_field":true,`, "flat_usage_out_of_scope": `"usage":{"reasoning_tokens":0},`,
		"metadata_array": `"metadata":{"unrelated_field":[]},`, "weight_version_array": `"metadata":{"weight_version":[]},`,
		"weight_versions_object": `"metadata":{"weight_versions":{}},`, "weight_versions_unknown": `"metadata":{"weight_versions":[{"unrelated_field":true}]},`,
	} {
		t.Run(name, func(t *testing.T) {
			for _, body := range []string{
				`{` + extra + `"id":"r","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}]}`,
				`data: {` + extra + `"id":"r","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"OK"},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n",
			} {
				var err error
				if bytes.HasPrefix([]byte(body), []byte("data:")) {
					_, _, err = NewBuiltinEngine().DecodeResponseStream(llmprotocol.OpenAIChatV1, []byte(body), llmprotocol.StreamContext{})
				} else {
					_, _, _, err = NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, []byte(body))
				}
				var protocolErr *llmprotocol.ProtocolError
				if !errors.As(err, &protocolErr) || protocolErr.Code != "invalid_upstream_json" || protocolErr.Category != llmprotocol.ErrorUpstreamUnavailable {
					t.Fatalf("expected unchanged upstream JSON error, got %v", err)
				}
			}
		})
	}
}

func TestSGLangExtensionsDoNotWidenCanonicalMetadata(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		_, _, _, err := engine.DecodeRequest(format, []byte(`{"model":"m","metadata":{"weight_versions":[]}}`))
		if cause := errors.Unwrap(err); cause == nil || !strings.Contains(cause.Error(), "cannot unmarshal array") {
			t.Fatalf("%s request metadata: %v", format, err)
		}
	}
	_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, []byte(`{"metadata":{"weight_versions":[]}}`))
	if cause := errors.Unwrap(err); cause == nil || !strings.Contains(cause.Error(), "cannot unmarshal array") {
		t.Fatalf("Responses response metadata: %v", err)
	}
}

// The stop stream is recorded with stop:["7"], omitting only flat reasoning
// usage (#4914). The buffered fixture reconstructs its content and terminal.
func TestSGLangMatchedStopTranslation(t *testing.T) {
	for _, tc := range []struct{ name, fields, finish, want string }{
		{"string", `"matched_stop":"7"`, "stop", `"stop_reason":"stop_sequence","stop_sequence":"7"`},
		{"precedence", `"matched_stop":"7","stop_reason":"OTHER"`, "stop", `"stop_reason":"stop_sequence","stop_sequence":"OTHER"`},
		{"precedence_token", `"matched_stop":"7","stop_reason":151645`, "stop", `"stop_reason":"end_turn","stop_sequence":null`},
		{"null", `"matched_stop":null`, "stop", `"stop_reason":"end_turn","stop_sequence":null`},
		{"zero", `"matched_stop":0`, "stop", `"stop_reason":"end_turn","stop_sequence":null`},
		{"token", `"matched_stop":151645`, "stop", `"stop_reason":"end_turn","stop_sequence":null`},
		{"length", `"matched_stop":"7"`, "length", `"stop_reason":"max_tokens","stop_sequence":null`},
	} {
		for _, mode := range []string{"buffered", "streamed", "repeated"} {
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				name := "stop-stream.sse"
				if mode == "buffered" {
					name = "stop-out.json"
				}
				body := bytes.ReplaceAll(sglangChatCapture(t, name), []byte(`"matched_stop":"7"`), []byte(tc.fields))
				body = bytes.ReplaceAll(body, []byte(`"finish_reason":"stop"`), []byte(`"finish_reason":"`+tc.finish+`"`))
				var out string
				if mode == "buffered" {
					result, err := NewBuiltinEngine().TranslateResponse(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, nil)
					if err != nil {
						t.Fatal(err)
					}
					out = string(result.Body)
				} else {
					if mode == "repeated" {
						for _, chunk := range bytes.SplitAfter(body, []byte("\n\n")) {
							if bytes.Contains(chunk, []byte(`"finish_reason":"`+tc.finish+`"`)) {
								body = bytes.Replace(body, chunk, append(bytes.Clone(chunk), chunk...), 1)
								break
							}
						}
					}
					out = translateMatchedStopStream(t, llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, string(body))
					if strings.Count(out, "event: message_stop\n") != 1 {
						t.Fatalf("expected one successful completion: %s", out)
					}
				}
				if !strings.Contains(out, tc.want) {
					t.Fatalf("want %s in %s", tc.want, out)
				}
			})
		}
	}
}

func TestSGLangWeightVersionKeysAreExact(t *testing.T) {
	body := sglangChatCapture(t, "out.json")
	for _, pair := range [][2]string{{"version", "Version"}, {"start", "START"}, {"end", "END"}, {"version", "unrelated_field"}} {
		t.Run(pair[1], func(t *testing.T) {
			invalid := bytes.ReplaceAll(body, []byte(`"`+pair[0]+`":`), []byte(`"`+pair[1]+`":`))
			_, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, invalid)
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != "invalid_upstream_json" {
				t.Fatalf("expected invalid_upstream_json for %s, got %v", pair[1], err)
			}
		})
	}
}
