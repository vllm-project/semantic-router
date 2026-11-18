package protocolcodec

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Valid JSON with a member of the wrong type, or a member the API does not
// define, is refused as OpenAI refuses it, naming the member; only a body
// that is not JSON is invalid_json.
func TestRequestFieldErrorsNameTheField(t *testing.T) {
	const chatMessages = `"messages":[{"role":"user","content":"hi"}]`
	for _, test := range []struct {
		name                 string
		format               llmprotocol.WireFormat
		body                 string
		code, param, message string
	}{
		{
			name: "chat decimal max_tokens", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m",` + chatMessages + `,"max_tokens":5.5}`, code: "invalid_type", param: "max_tokens",
			message: "Invalid type for 'max_tokens': expected an integer, but got a decimal number instead.",
		},
		{
			name: "chat string temperature", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m",` + chatMessages + `,"temperature":"cold"}`, code: "invalid_type", param: "temperature",
			message: "Invalid type for 'temperature': expected a number, but got a string instead.",
		},
		{
			name: "chat messages not an array", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m","messages":"hi"}`, code: "invalid_type", param: "messages",
			message: "Invalid type for 'messages': expected an array, but got a string instead.",
		},
		{
			name: "chat numeric role", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m","messages":[{"role":5,"content":"hi"}]}`, code: "invalid_type", param: "messages[].role",
			message: "Invalid type for 'messages[].role': expected a string, but got a number instead.",
		},
		{
			name: "chat output limit out of range", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m",` + chatMessages + `,"max_completion_tokens":9223372036854775808}`, code: "invalid_type", param: "max_completion_tokens",
			message: "Invalid type for 'max_completion_tokens': expected an integer, but got a number out of range instead.",
		},
		{
			name: "chat unknown member", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m",` + chatMessages + `,"foo":1}`, code: "unknown_parameter", param: "foo",
			message: "Unrecognized request argument supplied: foo",
		},
		{
			name: "responses decimal max_output_tokens", format: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","input":"hi","max_output_tokens":5.5}`, code: "invalid_type", param: "max_output_tokens",
			message: "Invalid type for 'max_output_tokens': expected an integer, but got a decimal number instead.",
		},
		{
			name: "responses unknown member", format: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","input":"hi","foo":1}`, code: "unknown_parameter", param: "foo",
			message: "Unrecognized request argument supplied: foo",
		},
		{
			name: "messages decimal max_tokens", format: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"m","max_tokens":5.5,` + chatMessages + `}`, code: "invalid_type", param: "max_tokens",
			message: "Invalid type for 'max_tokens': expected an integer, but got a decimal number instead.",
		},
		{
			name: "messages unknown member", format: llmprotocol.AnthropicMessagesV1,
			body: `{"model":"m","max_tokens":8,` + chatMessages + `,"foo":1}`, code: "unknown_parameter", param: "foo",
			message: "Unrecognized request argument supplied: foo",
		},
		{
			name: "chat syntax error", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m",` + chatMessages + `,"max_tokens":}`, code: "invalid_json",
			message: "request JSON is invalid",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeRequest(test.format, []byte(test.body))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) {
				t.Fatalf("err = %v, want a protocol error", err)
			}
			if protocolError.Category != llmprotocol.ErrorInvalidRequest || protocolError.Code != test.code ||
				protocolError.Parameter != test.param || protocolError.Message != test.message {
				t.Fatalf("error = %+v, want %s param %q: %s", protocolError, test.code, test.param, test.message)
			}
		})
	}
}

// The OpenAI envelope carries the code and the member; the Anthropic one,
// which has neither, says the same in its message.
func TestRequestFieldErrorsInEachClientEnvelope(t *testing.T) {
	_, _, _, decodeErr := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1,
		[]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":5.5}`))
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(decodeErr, &protocolError) {
		t.Fatalf("err = %v", decodeErr)
	}
	for format, want := range map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:      `{"error":{"type":"invalid_request_error","code":"invalid_type","message":"Invalid type for 'max_tokens': expected an integer, but got a decimal number instead.","param":"max_tokens"}}`,
		llmprotocol.OpenAIResponsesV1: `{"error":{"type":"invalid_request_error","code":"invalid_type","message":"Invalid type for 'max_tokens': expected an integer, but got a decimal number instead.","param":"max_tokens"}}`,
		llmprotocol.AnthropicMessagesV1: `{"type":"error","error":{"type":"invalid_request_error",` +
			`"message":"Invalid type for 'max_tokens': expected an integer, but got a decimal number instead."}}`,
	} {
		body, err := NewBuiltinEngine().EncodeError(format, protocolError)
		if err != nil || !jsonEquivalent(body, []byte(want)) {
			t.Fatalf("%s error = %s (%v), want %s", format, body, err, want)
		}
	}
}
