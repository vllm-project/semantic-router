package protocolcodec

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestProviderResponseEnumsAndDiscriminatorsAreClosed(t *testing.T) {
	tests := providerResponseValidationCases()
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeResponse(test.format, []byte(test.body))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

type providerResponseValidationCase struct {
	name   string
	format llmprotocol.WireFormat
	body   string
	code   string
}

func providerResponseValidationCases() []providerResponseValidationCase {
	chat := llmprotocol.OpenAIChatV1
	responses := llmprotocol.OpenAIResponsesV1
	anthropic := llmprotocol.AnthropicMessagesV1
	return []providerResponseValidationCase{
		{name: "chat object", format: chat, body: `{"id":"chat_1","object":"response","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, code: "invalid_chat_response_object"},
		{name: "chat finish reason", format: chat, body: `{"id":"chat_1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"future_reason"}]}`, code: "invalid_chat_finish_reason"},
		{name: "chat missing response role", format: chat, body: `{"id":"chat_1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"content":"ok"},"finish_reason":"stop"}]}`, code: "invalid_response_role"},
		{name: "chat incomplete error", format: chat, body: `{"error":{"type":"provider_error","message":""}}`, code: "invalid_chat_response_error"},
		{name: "responses object", format: responses, body: `{"id":"resp_1","object":"chat.completion","model":"m","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`, code: "invalid_responses_object"},
		{name: "responses status", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"future_status","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`, code: "invalid_responses_status"},
		{name: "responses buffered nonterminal status", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"in_progress","output":[]}`, code: "nonterminal_responses_resource"},
		{name: "responses incomplete reason", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"incomplete","incomplete_details":{"reason":"future_reason"},"output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`, code: "invalid_responses_incomplete_reason"},
		{name: "responses failed without error", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"failed","output":[]}`, code: "responses_error_required"},
		{name: "responses incomplete without details", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"incomplete","output":[]}`, code: "responses_incomplete_details_required"},
		{name: "responses details on completed resource", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"completed","incomplete_details":{"reason":"max_output_tokens"},"output":[]}`, code: "responses_incomplete_status"},
		{name: "responses error with completed status", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"completed","error":{"code":"provider_error","message":"failed"},"output":[]}`, code: "responses_error_status"},
		{name: "responses incomplete error", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"failed","error":{"code":"","message":"failed"},"output":[]}`, code: "invalid_responses_error"},
		{name: "responses output item status", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"future_status","content":[{"type":"output_text","text":"ok"}]}]}`, code: "invalid_responses_item_status"},
		{name: "responses missing output role", format: responses, body: `{"id":"resp_1","object":"response","model":"m","status":"completed","output":[{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"ok"}]}]}`, code: "invalid_response_role"},
		{name: "anthropic type", format: anthropic, body: `{"id":"msg_1","type":"response","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`, code: "invalid_anthropic_response_type"},
		{name: "anthropic role", format: anthropic, body: `{"id":"msg_1","type":"message","role":"user","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`, code: "invalid_anthropic_response_role"},
		{name: "anthropic stop reason", format: anthropic, body: `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"future_reason","usage":{"input_tokens":1,"output_tokens":1}}`, code: "invalid_anthropic_stop_reason"},
		{name: "anthropic incomplete error", format: anthropic, body: `{"error":{"type":"","message":"failed"}}`, code: "anthropic_transport_error_on_response_path"},
		{name: "anthropic missing matched stop sequence", format: anthropic, body: `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"stop_sequence","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`, code: "anthropic_stop_sequence_required"},
		{name: "anthropic sequence with wrong reason", format: anthropic, body: `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":"END","usage":{"input_tokens":1,"output_tokens":1}}`, code: "anthropic_stop_sequence_reason"},
		{name: "anthropic empty sequence with no reason", format: anthropic, body: `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":null,"stop_sequence":"","usage":{"input_tokens":1,"output_tokens":1}}`, code: "anthropic_stop_sequence_reason"},
		{name: "anthropic empty sequence with wrong reason", format: anthropic, body: `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","stop_sequence":"","usage":{"input_tokens":1,"output_tokens":1}}`, code: "anthropic_stop_sequence_reason"},
	}
}

func TestOfficialProviderResponseEnumInventoriesAreClosed(t *testing.T) {
	tests := []struct {
		name   string
		values []string
		valid  func(string) bool
	}{
		{
			name:   "Chat finish reason",
			values: []string{"stop", "length", "tool_calls", "content_filter", "function_call"},
			valid:  validChatFinishReason,
		},
		{
			name:   "Responses status",
			values: []string{"completed", "failed", "in_progress", "cancelled", "queued", "incomplete"},
			valid:  validResponsesStatus,
		},
		{
			name: "Anthropic stop reason",
			values: []string{
				"end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal",
				"model_context_window_exceeded",
			},
			valid: validAnthropicStopReason,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			seen := make(map[string]struct{}, len(test.values))
			for _, value := range test.values {
				if _, duplicate := seen[value]; duplicate {
					t.Fatalf("duplicate official enum %q", value)
				}
				seen[value] = struct{}{}
				if !test.valid(value) {
					t.Fatalf("official enum %q is not accepted", value)
				}
			}
			if test.valid("future_value") {
				t.Fatal("unknown future enum was accepted")
			}
		})
	}
}

func TestChatChoiceIndexesDefineCandidateOrder(t *testing.T) {
	body := []byte(`{
		"id":"chat_1","object":"chat.completion","model":"m",
		"choices":[
			{"index":1,"message":{"role":"assistant","content":"alternative"},"finish_reason":"stop"},
			{"index":0,"message":{"role":"assistant","content":"primary"},"finish_reason":"stop"}
		]
	}`)
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if got := response.Output[0].Content[0].Text; got != "primary" {
		t.Fatalf("primary choice = %q, want primary", got)
	}
	if len(response.Alternatives) != 1 || response.Alternatives[0][0].Content[0].Text != "alternative" {
		t.Fatalf("alternatives = %+v", response.Alternatives)
	}
}

func TestResponsesCancelledResourcePreservesTerminalReason(t *testing.T) {
	body := []byte(`{
		"id":"resp_cancelled","object":"response","model":"m","status":"cancelled",
		"output":[{"type":"message","id":"msg_1","role":"assistant","status":"incomplete","content":[{"type":"output_text","text":"partial"}]}]
	}`)
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StopReason != llmprotocol.StopCanceled || response.SourceStopReason != "cancelled" {
		t.Fatalf("cancelled response terminal = %q source=%q", response.StopReason, response.SourceStopReason)
	}
}

func TestChatChoiceIndexesRejectDuplicatesAndGaps(t *testing.T) {
	for _, choices := range []string{
		`[{"index":0,"message":{"role":"assistant","content":"first"},"finish_reason":"stop"},{"index":0,"message":{"role":"assistant","content":"duplicate"},"finish_reason":"stop"}]`,
		`[{"index":1,"message":{"role":"assistant","content":"gap"},"finish_reason":"stop"}]`,
	} {
		body := []byte(`{"id":"chat_1","object":"chat.completion","model":"m","choices":` + choices + `}`)
		_, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIChatV1, body)
		assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_chat_choice_index")
	}
}
