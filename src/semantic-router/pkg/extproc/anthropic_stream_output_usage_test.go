package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// An extended-thinking answer shorter than message_start's provisional output
// count reaches the client whole and ends with message_stop.
func TestAnthropicShortThinkingAnswerStreamCompletes(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
	}
	body := []byte(strings.Join([]string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0,\"output_tokens\":7}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-1\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"42\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":30,\"output_tokens_details\":{\"thinking_tokens\":26}}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}, ""))
	response := router.handleSemanticStreamingResponseBody(body, true, ctx)
	if ctx.StreamingAborted || !ctx.StreamingComplete {
		t.Fatalf("short thinking answer did not complete: aborted=%v complete=%v", ctx.StreamingAborted, ctx.StreamingComplete)
	}
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil &&
		strings.Contains(string(mutation.GetBody()), "event: error") {
		t.Fatalf("client stream ended in an error event: %s", mutation.GetBody())
	}
	if ctx.SemanticResponse == nil {
		t.Fatal("stream left no semantic response")
	}
	usage := ctx.SemanticResponse.Usage
	if usage.OutputTotal.Value == nil || *usage.OutputTotal.Value != 30 ||
		usage.OutputReasoning.Value == nil || *usage.OutputReasoning.Value != 26 {
		t.Fatalf("settled output usage = total %v reasoning %v, want 30 and 26", usage.OutputTotal.Value, usage.OutputReasoning.Value)
	}
}
