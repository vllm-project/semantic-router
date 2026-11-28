package extproc

import (
	"bytes"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

const extProcRedactedPayload = "EmwKAhgB...opaque=="

// A local redacted_thinking SSE fixture must keep its redacted scope through
// ExtProc observation and reconstruction. response() drops the scope if the
// stream state does not carry it, so a later replay re-encodes the opaque
// block as ordinary thinking and changes what is saved for cache and replay.
// This pins the whole path: observe -> response() -> EncodeResponseStream.
func TestExtProcRedactedThinkingKeepsScopeThroughResponse(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		TargetFormat: llmprotocol.AnthropicMessagesV1,
		RequestModel: "public-model",
		TraceContext: t.Context(),
	}
	if err := router.ensureSemanticResponseStream(ctx); err != nil {
		t.Fatalf("ensureSemanticResponseStream(): %v", err)
	}
	payload := []byte(strings.Join([]string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"response_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"source-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"" + extProcRedactedPayload + "\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}, ""))
	clientWire := pushExtProcStreamFixture(t, ctx, payload)
	// The live client stream stays redacted.
	if !bytes.Contains(clientWire.Bytes(), []byte(`"type":"redacted_thinking"`)) ||
		!bytes.Contains(clientWire.Bytes(), []byte(extProcRedactedPayload)) {
		t.Fatalf("live client stream lost the redacted block: %s", clientWire.Bytes())
	}

	semantic, err := ctx.SemanticStreamState.response()
	if err != nil {
		t.Fatalf("response(): %v", err)
	}
	var redactedScope bool
	for _, output := range semantic.Output {
		for _, block := range output.Content {
			if block.Reasoning == llmprotocol.ReasoningScopeRedacted {
				redactedScope = true
				if block.Signature != extProcRedactedPayload || block.Text != "" {
					t.Fatalf("reconstructed redacted block = %+v", block)
				}
			}
		}
	}
	if !redactedScope {
		t.Fatalf("response() dropped the redacted scope: %+v", semantic.Output)
	}

	// Re-encoding the reconstructed response as a stream must keep the opaque
	// block rather than turning it into an ordinary thinking block.
	engine := protocolcodec.NewBuiltinEngine()
	replayed, _, err := engine.EncodeResponseStream(
		llmprotocol.AnthropicMessagesV1, *semantic,
		llmprotocol.StreamContext{Context: t.Context(), PublicModel: semantic.Model},
	)
	if err != nil {
		t.Fatalf("EncodeResponseStream(): %v", err)
	}
	if !bytes.Contains(replayed, []byte(`"type":"redacted_thinking"`)) ||
		!bytes.Contains(replayed, []byte(extProcRedactedPayload)) {
		t.Fatalf("replayed stream re-encoded redacted state as ordinary thinking: %s", replayed)
	}
	if bytes.Contains(replayed, []byte(`{"signature":"`)) {
		t.Fatalf("replayed stream wrote a thinking signature instead of redacted data: %s", replayed)
	}
}
