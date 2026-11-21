package protocolcodec

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type responsesItemIDWant struct{ prefix, callID string }

// A Responses client sends the items of an answer back with their ids, and an
// OpenAI backend refuses an id whose prefix is not its type's. An answer from
// a Chat backend is one neutral item that holds reasoning, text and a tool
// call, which the encoder splits into several Responses items: each gets the
// prefix of its type, buffered and streamed, as does a custom tool call,
// which a stream starts as custom_tool_call, and a tool call keeps its
// call_id. Ids that already have their type's prefix, as an OpenAI backend's
// do, pass unchanged.
func TestResponsesItemsHaveTheIDPrefixOfTheirType(t *testing.T) {
	fromChat := llmprotocol.OutputItem{ID: "item_from_chat", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentReasoning, Text: "Look first."},
		{Kind: llmprotocol.ContentText, Text: "Looking."},
		{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call_ls", Name: "ls", Arguments: "{}"}},
	}}
	fromOpenAI := llmprotocol.OutputItem{ID: "msg_upstream", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Done."}}}
	custom := llmprotocol.OutputItem{ID: "item_custom", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
		{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{Kind: llmprotocol.ToolKindCustom, ID: "call_patch", Name: "patch", Arguments: "x"}},
	}}
	want := []responsesItemIDWant{{"rs_", ""}, {"msg_", ""}, {"fc_", "call_ls"}, {"msg_upstream", ""}, {"ctc_", "call_patch"}}
	engine := NewBuiltinEngine()

	buffered, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, responsesIDAnswer(fromChat, fromOpenAI, custom), llmprotocol.Envelope{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	fromBody, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, buffered.Body)
	if err != nil {
		t.Fatalf("decode %s: %v", buffered.Body, err)
	}
	requireResponsesItemIDs(t, "buffered", fromBody, want)

	streamed, _, err := engine.EncodeResponseStream(llmprotocol.OpenAIResponsesV1, responsesIDAnswer(fromChat, fromOpenAI, custom), llmprotocol.StreamContext{Context: context.Background()})
	if err != nil {
		t.Fatalf("encode stream: %v", err)
	}
	fromStream, _, err := engine.DecodeResponseStream(llmprotocol.OpenAIResponsesV1, streamed, llmprotocol.StreamContext{Context: context.Background()})
	if err != nil {
		t.Fatalf("decode stream: %v", err)
	}
	requireResponsesItemIDs(t, "streamed", fromStream, want)
	if custom := fromStream.Output[4].Content[0].ToolCall; custom.Kind != llmprotocol.ToolKindCustom || custom.Arguments != "x" {
		t.Fatalf("streamed custom tool call = %+v, want its kind and input", custom)
	}
}

func responsesIDAnswer(items ...llmprotocol.OutputItem) llmprotocol.Response {
	return llmprotocol.Response{
		Generation: 1, ID: "resp_answer", Model: "m", CreatedAt: time.Unix(1700000000, 0).UTC(),
		StopReason: llmprotocol.StopToolCall, Usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable}, Output: items,
	}
}

func requireResponsesItemIDs(t *testing.T, name string, answer llmprotocol.Response, want []responsesItemIDWant) {
	t.Helper()
	if len(answer.Output) != len(want) {
		t.Fatalf("%s: %d items %+v, want %d", name, len(answer.Output), answer.Output, len(want))
	}
	for index, item := range answer.Output {
		if !strings.HasPrefix(item.ID, want[index].prefix) || strings.HasPrefix(item.ID, "item_") {
			t.Fatalf("%s: item %d id = %q, want %s", name, index, item.ID, want[index].prefix)
		}
		if call := item.Content[0].ToolCall; want[index].callID != "" && (call == nil || call.ID != want[index].callID) {
			t.Fatalf("%s: item %d = %+v, want call_id %q kept", name, index, item, want[index].callID)
		}
	}
}
