package protocolcodec

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A translated stream names PublicModel by default. With PreferProviderModel it
// names the model the provider's stream reports, in every frame.
func TestTranslatedStreamNamesTheProviderModelWhenPreferred(t *testing.T) {
	upstream := []byte(strings.Join([]string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"resolved-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}, ""))
	for _, tc := range []struct {
		prefer      bool
		want, avoid string
	}{
		{false, "requested-model", "resolved-model"},
		{true, "resolved-model", "requested-model"},
	} {
		for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
			stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target, llmprotocol.StreamContext{
				Context: context.Background(), PublicModel: "requested-model", PreferProviderModel: tc.prefer,
			})
			if err != nil {
				t.Fatal(err)
			}
			frames, _, _, err := stream.Push(upstream)
			if err != nil {
				t.Fatal(err)
			}
			last, _, _, err := stream.Finalize(nil)
			if err != nil {
				t.Fatal(err)
			}
			body := bytes.Join(append(frames, last...), nil)
			if bytes.Count(body, []byte(tc.want)) == 0 || bytes.Contains(body, []byte(tc.avoid)) {
				t.Errorf("prefer=%v %s: want only %s\n%s", tc.prefer, target, tc.want, body)
			}
		}
	}
}

// A provider that names its model only after the stream has started leaves the
// whole stream, its frames and its decoded events, on PublicModel, so the
// client and the reconstructed reply agree.
func TestLateProviderModelLeavesTheStreamOnPublicModel(t *testing.T) {
	upstream := []byte(strings.Join([]string{
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"first \"},\"finish_reason\":null}]}\n\n",
		"data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"late-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"},\"finish_reason\":\"stop\"}]}\n\n",
		"data: [DONE]\n\n",
	}, ""))
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "requested-model", PreferProviderModel: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	frames, events, _, err := stream.Push(upstream)
	if err != nil {
		t.Fatal(err)
	}
	last, tail, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	body := bytes.Join(append(frames, last...), nil)
	if !bytes.Contains(body, []byte("requested-model")) || bytes.Contains(body, []byte("late-model")) {
		t.Errorf("client frames do not stay on the first model:\n%s", body)
	}
	for _, event := range append(events, tail...) {
		if event.Model != "" && event.Model != "requested-model" {
			t.Errorf("decoded %s names %q", event.Type, event.Model)
		}
	}
}
