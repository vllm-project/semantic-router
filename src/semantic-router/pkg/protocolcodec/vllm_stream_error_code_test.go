package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The same event shape after content has started, as vLLM sends for an error
// raised during generation (synthetic: written in the shape of vLLM's chunks
// and of its ErrorInfo, not captured).
const vllmStreamMidError = `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"Qwen/Qwen3.8-27B-FP8","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"logprobs":null,"finish_reason":null}]}

data: {"error": {"message": "Internal server error", "type": "InternalServerError", "param": null, "code": 500}}

data: [DONE]

`

const vllmGrammarMessage = "Grammar error: regex parse error:\n    [\n    ^\nerror: unclosed character class"

var vllmErrorTargets = []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1}

// vllmStreamGrammarError is captured from vLLM 0.30.0: a streamed Chat request
// whose json_schema holds an invalid regex. vLLM answers HTTP 200, sends its
// error as an SSE event with an integer code, then [DONE].
func vllmStreamGrammarError(t *testing.T) string {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "providers", "vllm-chat-stream-error-stream.sse"))
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

func translateVLLMErrorStream(t *testing.T, target llmprotocol.WireFormat, chunks ...string) string {
	return string(bytes.Join(vllmErrorStreamFrames(t, target, chunks...), nil))
}

// vllmErrorStreamFrames pushes every chunk through one stream and hands the
// last push error to Finalize. This is the codec's translation; which bytes a
// client receives for each read is up to the ExtProc handler (see
// processor_res_vllm_stream_error_test.go).
func vllmErrorStreamFrames(t *testing.T, target llmprotocol.WireFormat, chunks ...string) [][]byte {
	t.Helper()
	stream, err := NewBuiltinEngine().NewStreamWithMutation(llmprotocol.OpenAIChatV1, target, llmprotocol.StreamContext{
		Source: llmprotocol.OpenAIChatV1, Target: target, PublicModel: "m",
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var out [][]byte
	var streamErr error
	for _, chunk := range chunks {
		frames, _, _, pushErr := stream.Push([]byte(chunk))
		out = append(out, frames...)
		if pushErr != nil {
			streamErr = pushErr
		}
	}
	frames, _, _, _ := stream.Finalize(streamErr)
	return append(out, frames...)
}

func decodeVLLMErrorFrame(t *testing.T, frame []byte) (sseFrame, map[string]json.RawMessage) {
	t.Helper()
	parsed, err := parseSSEFrame(frame, llmprotocol.DefaultPolicy().Limits.SSEFrameBytes)
	if err != nil {
		t.Fatalf("client error frame is invalid: %v: %q", err, frame)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(parsed.Data, &object); err != nil {
		t.Fatalf("client error payload is invalid JSON: %v: %s", err, parsed.Data)
	}
	return parsed, object
}

// Translated, the stream ends on one canonical error frame with vLLM's status
// and message: 400 becomes invalid_request_error, 500 the format's server error.
func TestVLLMStreamErrorEventFrames(t *testing.T) {
	cases := map[string]struct{ body, message, code, chatType, messagesType string }{
		"grammar error before content": {vllmStreamGrammarError(t), vllmGrammarMessage, "400", "invalid_request_error", "invalid_request_error"},
		"error after content":          {vllmStreamMidError, "Internal server error", "500", "server_error", "api_error"},
	}
	for name, tc := range cases {
		for _, target := range vllmErrorTargets {
			t.Run(name+"/"+string(target), func(t *testing.T) {
				frames := vllmErrorStreamFrames(t, target, tc.body)
				if len(frames) == 0 {
					t.Fatal("client stream is empty")
				}
				wire := bytes.Join(frames, nil)
				assertMidstreamProtocolTerminal(t, target, wire)
				if head, _, _ := strings.Cut(tc.message, "\n"); bytes.Count(wire, []byte(head)) != 1 {
					t.Fatalf("want exactly one error frame carrying %q:\n%s", head, wire)
				}
				parsed, object := decodeVLLMErrorFrame(t, frames[len(frames)-1])
				switch target {
				case llmprotocol.OpenAIChatV1:
					assertChatStreamErrorPayload(t, parsed, object, tc.chatType, tc.code, tc.message, "")
				case llmprotocol.AnthropicMessagesV1:
					assertAnthropicStreamErrorPayload(t, parsed, object, tc.messagesType, tc.message)
				default:
					assertResponsesStreamErrorPayload(t, llmprotocol.FailureTransport, parsed, object, tc.code, tc.message, "")
				}
			})
		}
	}
}

// The codec gets the stream in arbitrary pieces. Where they split, two ways or
// one byte at a time, must not change its translation.
func TestVLLMStreamErrorEventSurvivesEverySplit(t *testing.T) {
	for _, tc := range []struct{ body, message string }{
		{vllmStreamGrammarError(t), "Grammar error: regex parse error"},
		{vllmStreamMidError, "Internal server error"},
	} {
		body := tc.body
		for _, target := range vllmErrorTargets {
			whole := translateVLLMErrorStream(t, target, body)
			if !strings.Contains(whole, tc.message) {
				t.Fatalf("%s: want vLLM's message %q:\n%s", target, tc.message, whole)
			}
			if bytewise := translateVLLMErrorStream(t, target, strings.Split(body, "")...); bytewise != whole {
				t.Fatalf("%s one byte at a time:\n%s\nwant:\n%s", target, bytewise, whole)
			}
			for at := 1; at < len(body); at++ {
				if split := translateVLLMErrorStream(t, target, body[:at], body[at:]); split != whole {
					t.Fatalf("%s split at byte %d:\n%s\nwant:\n%s", target, at, split, whole)
				}
			}
		}
	}
}

// The ExtProc handler takes a Chat client's error frame as the last frame of
// the batch that decoded the failure. Pin that for Chat to Chat: in every
// split, that batch ends on its one error frame and nothing is encoded after.
func TestChatStreamFailureEndsItsBatch(t *testing.T) {
	for _, body := range []string{vllmStreamGrammarError(t), vllmStreamMidError} {
		for at := 1; at < len(body); at++ {
			stream, err := NewBuiltinEngine().NewStreamWithMutation(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
				Source: llmprotocol.OpenAIChatV1, Target: llmprotocol.OpenAIChatV1, PublicModel: "m",
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			failed := false
			check := func(frames [][]byte, events []llmprotocol.Event) {
				t.Helper()
				if failed && len(frames) > 0 {
					t.Fatalf("split at byte %d: frames after the failure: %q", at, frames)
				}
				for _, event := range events {
					if event.Type != llmprotocol.EventResponseFailed {
						continue
					}
					failed = true
					errorFrames := 0
					for _, frame := range frames {
						if _, object := decodeVLLMErrorFrame(t, frame); object["error"] != nil {
							errorFrames++
						}
					}
					if len(frames) == 0 || errorFrames != 1 {
						t.Fatalf("split at byte %d: want one error frame in the failure's batch, got %d of %d: %q", at, errorFrames, len(frames), frames)
					}
					if _, object := decodeVLLMErrorFrame(t, frames[len(frames)-1]); object["error"] == nil {
						t.Fatalf("split at byte %d: the failure's batch does not end on its error frame: %q", at, frames)
					}
				}
			}
			// As in vllmErrorStreamFrames, a push error (data after the
			// failure) goes to Finalize; the push's frames still count.
			var streamErr error
			for _, chunk := range []string{body[:at], body[at:]} {
				frames, events, _, pushErr := stream.Push([]byte(chunk))
				check(frames, events)
				if pushErr != nil {
					streamErr = pushErr
				}
			}
			frames, events, _, _ := stream.Finalize(streamErr)
			check(frames, events)
			if !failed {
				t.Fatalf("split at byte %d: no failure event", at)
			}
		}
	}
}

// Only the integral HTTP status is newly accepted. Every other code decodes
// as before, on the stream event and on the buffered Chat error envelope.
func TestChatErrorCodeShapes(t *testing.T) {
	for _, tc := range []struct{ name, field, want string }{
		{"integer status", `, "code": 400`, "400"},
		{"highest status", `, "code": 599`, "599"},
		{"string status", `, "code": "400"`, "400"},
		{"string code", `, "code": "rate_limit_exceeded"`, "rate_limit_exceeded"},
		{"null", `, "code": null`, ""},
		{"absent", ``, ""},
		{"non-error integer", `, "code": 200`, "invalid_upstream_json"},
		{"below status range", `, "code": 399`, "invalid_upstream_json"},
		{"above status range", `, "code": 600`, "invalid_upstream_json"},
		{"decimal literal", `, "code": 400.0`, "invalid_upstream_json"},
		{"fraction", `, "code": 400.5`, "invalid_upstream_json"},
		{"boolean", `, "code": true`, "invalid_upstream_json"},
		{"object", `, "code": {}`, "invalid_upstream_json"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			event := `{"error": {"message": "m", "type": "BadRequestError", "param": null` + tc.field + `}}`
			frames := vllmErrorStreamFrames(t, llmprotocol.OpenAIChatV1, "data: "+event+"\n\n")
			_, object := decodeVLLMErrorFrame(t, frames[len(frames)-1])
			var streamed struct {
				Code *string `json:"code"`
			}
			if err := json.Unmarshal(object["error"], &streamed); err != nil || (streamed.Code == nil) != (tc.want == "") ||
				streamed.Code != nil && *streamed.Code != tc.want {
				t.Errorf("stream event: client error = %s, want code %q", object["error"], tc.want)
			}
			if got := bufferedChatErrorCode(event); got != tc.want {
				t.Errorf("buffered envelope: code %q, want %q", got, tc.want)
			}
		})
	}
}

// bufferedChatErrorCode is the error code a buffered Chat body decodes to, or
// the decode error's code when the body is rejected.
func bufferedChatErrorCode(body string) string {
	response, _, _, err := OpenAIChatCodec{}.DecodeResponse([]byte(body), llmprotocol.DefaultPolicy())
	var protocolError *llmprotocol.ProtocolError
	switch {
	case errors.As(err, &protocolError):
		return protocolError.Code
	case err != nil:
		return err.Error()
	case response.Error == nil:
		return "no error decoded"
	}
	return response.Error.Code
}
