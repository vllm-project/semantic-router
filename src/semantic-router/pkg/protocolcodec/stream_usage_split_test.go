package protocolcodec

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// vllmCacheSplitStream is the shape of vLLM's Messages stream with
// --enable-prompt-tokens-details: message_start counts the whole prompt as
// input_tokens, and message_delta splits the same prompt into uncached input,
// cache reads and cache writes.
func vllmCacheSplitStream(deltaUsage string) []byte {
	return []byte("event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":16712,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":` + deltaUsage + `}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n")
}

func decodeCacheSplitStream(deltaUsage string) (llmprotocol.Response, error) {
	resp, _, err := NewBuiltinEngine().DecodeResponseStream(llmprotocol.AnthropicMessagesV1, vllmCacheSplitStream(deltaUsage),
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "m"})
	return resp, err
}

func TestAnthropicStreamAcceptsLaterCacheSplitOfSameInput(t *testing.T) {
	resp, err := decodeCacheSplitStream(`{"input_tokens":248,"output_tokens":736,"cache_creation_input_tokens":784,"cache_read_input_tokens":15680}`)
	if err != nil {
		t.Fatalf("DecodeResponseStream error: %v", err)
	}
	want := map[string]int64{"uncached": 248, "cache read": 15680, "cache write": 784, "input total": 16712}
	got := map[string]*int64{
		"uncached": resp.Usage.InputUncached.Value, "cache read": resp.Usage.InputCacheRead.Value,
		"cache write": resp.Usage.InputCacheWrite.Value, "input total": resp.Usage.InputTotal.Value,
	}
	for name, value := range got {
		if value == nil || *value != want[name] {
			t.Fatalf("%s = %v, want %d", name, value, want[name])
		}
	}
}

func TestAnthropicStreamStillRejectsSmallerInputTotal(t *testing.T) {
	_, err := decodeCacheSplitStream(`{"input_tokens":248,"output_tokens":736,"cache_read_input_tokens":15000}`)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "usage_decreased")
}
