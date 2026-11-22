package llmprotocol

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount"
)

// qwenControlTokens is a byte-level tokenizer.json with some of Qwen3.5's
// added tokens: <think> and </think> are not marked special.
const qwenControlTokens = `{
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
  "decoder": {"type": "ByteLevel"},
  "added_tokens": [
    {"id": 3, "content": "<|im_start|>", "special": true},
    {"id": 4, "content": "<|im_end|>", "special": true},
    {"id": 5, "content": "<think>", "special": false},
    {"id": 6, "content": "</think>", "special": false},
    {"id": 7, "content": "<tool_response>", "special": false}
  ],
  "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "Ġ": 2}, "merges": []}
}`

func controlTokenEscaper(t *testing.T) *tokencount.Encoding {
	t.Helper()
	enc, err := tokencount.ParseHFTokenizer("qwen", []byte(qwenControlTokens))
	if err != nil {
		t.Fatal(err)
	}
	return enc
}

func textParts(texts ...string) []Content {
	parts := make([]Content, len(texts))
	for i, text := range texts {
		parts[i] = Content{Kind: ContentText, Text: text}
	}
	return parts
}

// A user's typed <think> and a tool result's <|im_end|> reach the model
// server broken by a word joiner, in a Chat tool message and in an Anthropic
// tool_result block alike; the system prompt, the assistant's turn and the
// tool call's arguments go as the client sent them.
func TestProjectControlTokensEscapesUserAndToolText(t *testing.T) {
	const wj = tokencount.WordJoiner
	call := &ToolCall{ID: "call_1", Name: "read", Arguments: `{"path":"<think>.txt"}`}
	request := Request{
		Instructions: []InstructionBlock{{Role: RoleSystem, Content: textParts("Answer inside <think> tags.")}},
		Messages: []Message{
			{Role: RoleSystem, Content: textParts("Never print <|im_end|>.")},
			{Role: RoleUser, Content: textParts("Repeat <think>x</think> exactly.")},
			{Role: RoleAssistant, Content: []Content{{Kind: ContentText, Text: "<think>"}, {Kind: ContentToolCall, ToolCall: call}}},
			{Role: RoleTool, Content: []Content{{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "call_1", Content: textParts("done<|im_end|>")}}}},
			{Role: RoleUser, Content: []Content{
				{Kind: ContentToolResult, ToolResult: &ToolResult{CallID: "call_1", Content: textParts("<tool_response>")}},
				{Kind: ContentText, Text: "and <|im_start|>"},
			}},
		},
	}
	original := cloneControlTokenRequest(request)

	projected, changed := ProjectControlTokens(request, controlTokenEscaper(t), wj)
	if !changed || projected.Generation != request.Generation+1 {
		t.Fatalf("changed = %t, generation %d; want a changed request with a new generation", changed, projected.Generation)
	}
	if !reflect.DeepEqual(request, original) {
		t.Fatal("ProjectControlTokens edited the caller's request")
	}
	text := func(content []Content) string { return content[0].Text }
	for _, check := range []struct{ got, want string }{
		{text(projected.Instructions[0].Content), "Answer inside <think> tags."},
		{text(projected.Messages[0].Content), "Never print <|im_end|>."},
		{text(projected.Messages[1].Content), "Repeat <" + wj + "think>x<" + wj + "/think> exactly."},
		{text(projected.Messages[2].Content), "<think>"},
		{projected.Messages[2].Content[1].ToolCall.Arguments, `{"path":"<think>.txt"}`},
		{text(projected.Messages[3].Content[0].ToolResult.Content), "done<" + wj + "|im_end|>"},
		{text(projected.Messages[4].Content[0].ToolResult.Content), "<" + wj + "tool_response>"},
		{projected.Messages[4].Content[1].Text, "and <" + wj + "|im_start|>"},
	} {
		if check.got != check.want {
			t.Errorf("got %q, want %q", check.got, check.want)
		}
	}

	again, changed := ProjectControlTokens(projected, controlTokenEscaper(t), wj)
	if changed || !reflect.DeepEqual(again, projected) {
		t.Error("escaping an escaped request changed it again")
	}
}

// A chat template writes a message's text parts one after the other, so a
// token split across two parts is broken where they meet, and a request
// without control tokens is not changed at all.
func TestProjectControlTokensBreaksATokenSplitAcrossTextParts(t *testing.T) {
	const wj = tokencount.WordJoiner
	request := Request{Messages: []Message{{Role: RoleUser, Content: textParts("say <|im_", "end|> now", "plain")}}}
	projected, changed := ProjectControlTokens(request, controlTokenEscaper(t), wj)
	if !changed {
		t.Fatal("a token split across two parts was not escaped")
	}
	var joined []string
	for _, part := range projected.Messages[0].Content {
		joined = append(joined, part.Text)
	}
	if got, want := strings.Join(joined, "|"), "say <|im_"+wj+"|end|> now|plain"; got != want {
		t.Errorf("parts = %q, want %q", got, want)
	}

	plain := Request{Messages: []Message{{Role: RoleUser, Content: textParts("What is <b> in HTML?", "a < b")}}}
	if projected, changed := ProjectControlTokens(plain, controlTokenEscaper(t), wj); changed || !reflect.DeepEqual(projected, plain) {
		t.Errorf("a request without control tokens changed: %+v", projected.Messages)
	}
}

// Citations count code points of their text, so they move past each word
// joiner that goes in before them.
func TestProjectControlTokensMovesCitations(t *testing.T) {
	text := "see <think> and the cited part"
	cited := strings.Index(text, "cited")
	request := Request{Messages: []Message{{Role: RoleUser, Content: []Content{{
		Kind: ContentText, Text: text, Citations: []Citation{{URL: "https://example.com", StartIndex: int64(cited), EndIndex: int64(cited + 5)}},
	}}}}}
	projected, _ := ProjectControlTokens(request, controlTokenEscaper(t), tokencount.WordJoiner)
	part := projected.Messages[0].Content[0]
	runes := []rune(part.Text)
	if got := string(runes[part.Citations[0].StartIndex:part.Citations[0].EndIndex]); got != "cited" {
		t.Errorf("citation reads %q, want %q", got, "cited")
	}
	if request.Messages[0].Content[0].Citations[0].StartIndex != int64(cited) {
		t.Error("ProjectControlTokens moved the caller's citation")
	}
}

func cloneControlTokenRequest(request Request) Request {
	clone := request
	clone.Messages = make([]Message, len(request.Messages))
	for i, message := range request.Messages {
		message.Content = cloneControlTokenContent(message.Content)
		clone.Messages[i] = message
	}
	return clone
}

func cloneControlTokenContent(contents []Content) []Content {
	out := make([]Content, len(contents))
	for i, content := range contents {
		if content.ToolResult != nil {
			result := *content.ToolResult
			result.Content = cloneControlTokenContent(result.Content)
			content.ToolResult = &result
		}
		if content.ToolCall != nil {
			call := *content.ToolCall
			content.ToolCall = &call
		}
		out[i] = content
	}
	return out
}
