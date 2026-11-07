package extproc

import (
	"bytes"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The current Anthropic API sends usage.speed in every message_start. The
// stream completes, and a same-format client receives the provider's bytes,
// speed included.
func TestSameFormatMessagesStreamForwardsUsageSpeed(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
	}
	body := []byte(strings.Join([]string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1,\"service_tier\":\"standard\",\"inference_geo\":\"global\",\"speed\":\"standard\"}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":4}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}, ""))
	response := router.handleSemanticStreamingResponseBody(body, true, ctx)
	if ctx.StreamingAborted || !ctx.StreamingComplete {
		t.Fatalf("stream with usage.speed did not complete: aborted=%v complete=%v", ctx.StreamingAborted, ctx.StreamingComplete)
	}
	if mutation := response.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		if got := mutation.GetBody(); !bytes.Equal(got, body) {
			t.Fatalf("same-format stream should forward the provider's bytes, speed included:\n got: %s\nwant: %s", got, body)
		}
	}
	if ctx.SemanticResponse == nil || ctx.SemanticResponse.Usage.OutputTotal.Value == nil ||
		*ctx.SemanticResponse.Usage.OutputTotal.Value != 4 {
		t.Fatalf("settled usage changed: %+v", ctx.SemanticResponse)
	}
}
