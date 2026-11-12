package protocolcodec

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Chat stream whose client asked for usage ends as OpenAI's streams end:
// the finish chunk, a usage-only chunk whose choices are an empty list, then
// [DONE]. Clients find the usage chunk by its empty choices, and strict ones
// refuse a chunk without them. The Router's own decoder reads it back.
func TestChatStreamEndsWithAUsageChunkWithEmptyChoices(t *testing.T) {
	prompt, completion := int64(13), int64(7)
	response := llmprotocol.Response{
		Generation: 1, ID: "chatcmpl-usage", Model: "public-model", StopReason: llmprotocol.StopEndTurn,
		Output: []llmprotocol.OutputItem{{
			ID: "item_1", Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "done"}},
		}},
		Usage: llmprotocol.Usage{
			State:       llmprotocol.UsageAvailable,
			InputTotal:  llmprotocol.TokenCount{Value: &prompt, Provenance: llmprotocol.UsageAuthoritative},
			OutputTotal: llmprotocol.TokenCount{Value: &completion, Provenance: llmprotocol.UsageAuthoritative},
		},
	}
	engine := NewBuiltinEngine()
	body, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIChatV1, response, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: response.Model,
		Options: llmprotocol.StreamOptions{IncludeUsage: boolPointer(true)},
	})
	if err != nil {
		t.Fatal(err)
	}
	frames := bytes.Split(bytes.TrimSpace(body), []byte("\n\n"))
	if len(frames) < 3 || string(frames[len(frames)-1]) != "data: [DONE]" {
		t.Fatalf("the stream does not end with [DONE]:\n%s", body)
	}
	if !bytes.Contains(frames[len(frames)-3], []byte(`"finish_reason":"stop"`)) {
		t.Fatalf("the finish chunk does not precede the usage chunk:\n%s", body)
	}
	var usageChunk map[string]json.RawMessage
	if err = json.Unmarshal(bytes.TrimPrefix(frames[len(frames)-2], []byte("data: ")), &usageChunk); err != nil {
		t.Fatal(err)
	}
	if string(usageChunk["choices"]) != "[]" {
		t.Fatalf("the usage-only chunk's choices = %s, want []:\n%s", usageChunk["choices"], body)
	}
	if string(usageChunk["usage"]) != `{"prompt_tokens":13,"completion_tokens":7,"total_tokens":20}` {
		t.Fatalf("the usage-only chunk's usage = %s:\n%s", usageChunk["usage"], body)
	}

	stream, err := engine.NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: response.Model,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, events, _, err := stream.Push(body)
	if err != nil {
		t.Fatalf("the Router does not read its own stream back: %v\n%s", err, body)
	}
	_, final, _, err := stream.Finalize(nil)
	if err != nil {
		t.Fatal(err)
	}
	var usage *llmprotocol.Usage
	for _, event := range append(events, final...) {
		if event.Usage != nil {
			usage = event.Usage
		}
	}
	if usage == nil || usage.InputTotal.Value == nil || *usage.InputTotal.Value != prompt ||
		usage.OutputTotal.Value == nil || *usage.OutputTotal.Value != completion {
		t.Fatalf("decoded usage = %+v, want %d in and %d out", usage, prompt, completion)
	}
}
