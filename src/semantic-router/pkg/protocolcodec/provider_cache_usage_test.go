package protocolcodec

import (
	"context"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestProviderPromptCacheUsagePreservesBufferedFieldPresence(t *testing.T) {
	one := int64(1)
	zero := int64(0)
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		body   []byte
		read   *int64
		write  *int64
	}{
		{
			name:   "responses missing",
			format: llmprotocol.OpenAIResponsesV1,
			body:   responsesCacheUsageBody(`{"input_tokens":4,"output_tokens":2,"total_tokens":6}`),
		},
		{
			name:   "responses partial",
			format: llmprotocol.OpenAIResponsesV1,
			body:   responsesCacheUsageBody(`{"input_tokens":4,"input_tokens_details":{"cached_tokens":1},"output_tokens":2,"total_tokens":6}`),
			read:   &one,
		},
		{
			name:   "responses explicit zero",
			format: llmprotocol.OpenAIResponsesV1,
			body:   responsesCacheUsageBody(`{"input_tokens":4,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":2,"total_tokens":6}`),
			read:   &zero,
			write:  &zero,
		},
		{
			name:   "anthropic missing",
			format: llmprotocol.AnthropicMessagesV1,
			body:   anthropicCacheUsageBody(`{"input_tokens":4,"output_tokens":2}`),
		},
		{
			name:   "anthropic partial",
			format: llmprotocol.AnthropicMessagesV1,
			body:   anthropicCacheUsageBody(`{"input_tokens":4,"cache_read_input_tokens":1,"output_tokens":2}`),
			read:   &one,
		},
		{
			name:   "anthropic explicit zero",
			format: llmprotocol.AnthropicMessagesV1,
			body:   anthropicCacheUsageBody(`{"input_tokens":4,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":2}`),
			read:   &zero,
			write:  &zero,
		},
	}

	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, _, _, err := engine.DecodeResponse(test.format, test.body)
			if err != nil {
				t.Fatal(err)
			}
			assertCacheTokenCount(t, "read", response.Usage.InputCacheRead, test.read)
			assertCacheTokenCount(t, "write", response.Usage.InputCacheWrite, test.write)
		})
	}
}

func TestProviderPromptCacheUsagePreservesStreamingFieldPresence(t *testing.T) {
	one := int64(1)
	zero := int64(0)
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		stream []byte
		read   *int64
		write  *int64
	}{
		{
			name:   "responses missing",
			format: llmprotocol.OpenAIResponsesV1,
			stream: responsesCacheUsageStream(`{"input_tokens":4,"output_tokens":2,"total_tokens":6}`),
		},
		{
			name:   "responses partial",
			format: llmprotocol.OpenAIResponsesV1,
			stream: responsesCacheUsageStream(`{"input_tokens":4,"input_tokens_details":{"cached_tokens":1},"output_tokens":2,"total_tokens":6}`),
			read:   &one,
		},
		{
			name:   "responses explicit zero",
			format: llmprotocol.OpenAIResponsesV1,
			stream: responsesCacheUsageStream(`{"input_tokens":4,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":2,"total_tokens":6}`),
			read:   &zero,
			write:  &zero,
		},
		{
			name:   "anthropic missing",
			format: llmprotocol.AnthropicMessagesV1,
			stream: anthropicCacheUsageStream(nil, nil),
		},
		{
			name:   "anthropic partial",
			format: llmprotocol.AnthropicMessagesV1,
			stream: anthropicCacheUsageStream(&one, nil),
			read:   &one,
		},
		{
			name:   "anthropic explicit zero",
			format: llmprotocol.AnthropicMessagesV1,
			stream: anthropicCacheUsageStream(&zero, &zero),
			read:   &zero,
			write:  &zero,
		},
	}

	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, _, err := engine.DecodeResponseStream(
				test.format,
				test.stream,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
			)
			if err != nil {
				t.Fatal(err)
			}
			assertCacheTokenCount(t, "read", response.Usage.InputCacheRead, test.read)
			assertCacheTokenCount(t, "write", response.Usage.InputCacheWrite, test.write)
		})
	}
}

func TestProviderPromptCacheUsagePreservesPresenceAcrossProtocolTranslation(t *testing.T) {
	one := int64(1)
	zero := int64(0)
	tests := []struct {
		name      string
		source    llmprotocol.WireFormat
		target    llmprotocol.WireFormat
		body      []byte
		wantRead  *int64
		wantWrite *int64
	}{
		{
			name:   "responses to anthropic missing",
			source: llmprotocol.OpenAIResponsesV1,
			target: llmprotocol.AnthropicMessagesV1,
			body:   responsesCacheUsageBody(`{"input_tokens":4,"output_tokens":2,"total_tokens":6}`),
		},
		{
			name:     "anthropic to responses partial",
			source:   llmprotocol.AnthropicMessagesV1,
			target:   llmprotocol.OpenAIResponsesV1,
			body:     anthropicCacheUsageBody(`{"input_tokens":4,"cache_read_input_tokens":1,"output_tokens":2}`),
			wantRead: &one,
		},
		{
			name:      "responses to anthropic explicit zero",
			source:    llmprotocol.OpenAIResponsesV1,
			target:    llmprotocol.AnthropicMessagesV1,
			body:      responsesCacheUsageBody(`{"input_tokens":4,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":2,"total_tokens":6}`),
			wantRead:  &zero,
			wantWrite: &zero,
		},
	}

	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			translated, err := engine.TranslateResponse(test.source, test.target, test.body, nil)
			if err != nil {
				t.Fatal(err)
			}
			assertCacheTokenCount(t, "read", translated.Response.Usage.InputCacheRead, test.wantRead)
			assertCacheTokenCount(t, "write", translated.Response.Usage.InputCacheWrite, test.wantWrite)
		})
	}
}

func assertCacheTokenCount(t *testing.T, name string, actual llmprotocol.TokenCount, want *int64) {
	t.Helper()
	if want == nil {
		if actual.Value != nil || actual.Provenance != llmprotocol.UsageUnknown {
			t.Fatalf("%s cache count = %+v, want unknown", name, actual)
		}
		return
	}
	if actual.Value == nil || *actual.Value != *want || actual.Provenance != llmprotocol.UsageAuthoritative {
		t.Fatalf("%s cache count = %+v, want authoritative %d", name, actual, *want)
	}
}

func responsesCacheUsageBody(usage string) []byte {
	return []byte(`{"id":"resp_cache","object":"response","model":"provider-model","status":"completed","output":[{"type":"message","id":"item_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":` + usage + `}`)
}

func anthropicCacheUsageBody(usage string) []byte {
	return []byte(`{"id":"msg_cache","type":"message","role":"assistant","model":"provider-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,"usage":` + usage + `}`)
}

func responsesCacheUsageStream(usage string) []byte {
	return []byte("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_cache\",\"object\":\"response\",\"model\":\"provider-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
		"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"item_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n" +
		"event: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":2,\"output_index\":0,\"item_id\":\"item_1\",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}\n\n" +
		"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":3,\"output_index\":0,\"item_id\":\"item_1\",\"content_index\":0,\"delta\":\"done\"}\n\n" +
		"event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"sequence_number\":4,\"output_index\":0,\"item_id\":\"item_1\",\"content_index\":0,\"text\":\"done\"}\n\n" +
		"event: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"sequence_number\":5,\"output_index\":0,\"item_id\":\"item_1\",\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"done\",\"annotations\":[]}}\n\n" +
		"event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":6,\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"item_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"done\",\"annotations\":[]}]}}\n\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":7,\"response\":" + usageResource(usage) + "}\n\n")
}

func usageResource(usage string) string {
	return `{"id":"resp_cache","object":"response","model":"provider-model","status":"completed","output":[{"type":"message","id":"item_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":` + usage + `}`
}

func anthropicCacheUsageStream(read, write *int64) []byte {
	cacheFields := ""
	if read != nil {
		cacheFields += fmt.Sprintf(`,"cache_read_input_tokens":%d`, *read)
	}
	if write != nil {
		cacheFields += fmt.Sprintf(`,"cache_creation_input_tokens":%d`, *write)
	}
	return []byte(
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_cache\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":4,\"output_tokens\":0" + cacheFields + "}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"done\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":4,\"output_tokens\":2" + cacheFields + "}}\n\n" +
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	)
}
