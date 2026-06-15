package extproc

import (
	"strings"
	"testing"
)

func TestExtractContentFastAnthropic_MissingModel(t *testing.T) {
	body := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	if _, err := extractContentFastAnthropic(body); err == nil {
		t.Fatal("expected error for missing model field")
	}
}

func TestExtractContentFastAnthropic_StringContentAndSystem(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-7",
		"system": "you are helpful",
		"messages": [
			{"role": "user", "content": "hello"}
		]
	}`)
	got, err := extractContentFastAnthropic(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Model != "claude-opus-4-7" {
		t.Fatalf("model: got %q, want claude-opus-4-7", got.Model)
	}
	if got.UserContent != "hello" {
		t.Fatalf("user content: got %q, want hello", got.UserContent)
	}
	if got.UserMessageCount != 1 {
		t.Fatalf("user count: got %d, want 1", got.UserMessageCount)
	}
	if got.SystemMessageCount != 1 {
		t.Fatalf("system count: got %d, want 1", got.SystemMessageCount)
	}
	if len(got.NonUserMessages) != 1 || got.NonUserMessages[0] != "you are helpful" {
		t.Fatalf("non-user messages: got %+v", got.NonUserMessages)
	}
}

func TestExtractContentFastAnthropic_ArraySystem(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-7",
		"system": [
			{"type": "text", "text": "block one"},
			{"type": "text", "text": "block two"}
		],
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	got, err := extractContentFastAnthropic(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Array system is still one logical system message; both texts join.
	if got.SystemMessageCount != 1 {
		t.Fatalf("system count: got %d, want 1", got.SystemMessageCount)
	}
	if len(got.NonUserMessages) != 1 {
		t.Fatalf("expected one joined non-user message, got %+v", got.NonUserMessages)
	}
	if !strings.Contains(got.NonUserMessages[0], "block one") || !strings.Contains(got.NonUserMessages[0], "block two") {
		t.Fatalf("expected both blocks joined, got %q", got.NonUserMessages[0])
	}
}

func TestExtractContentFastAnthropic_ArrayUserContentJoinsText(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-7",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "part one"},
				{"type": "text", "text": "part two"}
			]
		}]
	}`)
	got, err := extractContentFastAnthropic(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(got.UserContent, "part one") || !strings.Contains(got.UserContent, "part two") {
		t.Fatalf("expected both parts in user content, got %q", got.UserContent)
	}
}

func TestExtractContentFastAnthropic_StreamFlag(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-7",
		"stream": true,
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	got, err := extractContentFastAnthropic(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !got.Stream {
		t.Fatal("expected stream=true")
	}
}

func TestExtractContentFastAnthropic_ImageBase64(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-7",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "what is this"},
				{
					"type": "image",
					"source": {
						"type": "base64",
						"media_type": "image/png",
						"data": "iVBORw0KGgo="
					}
				}
			]
		}]
	}`)
	got, err := extractContentFastAnthropic(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "data:image/png;base64,iVBORw0KGgo="
	if got.FirstImageURL != want {
		t.Fatalf("first image url: got %q, want %q", got.FirstImageURL, want)
	}
}

func TestExtractContentFastAnthropic_ImageURLNotSurfaced(t *testing.T) {
	// URL and file_id image sources are not surfaced through the fast path
	// to preserve SSRF safety; the full inbound parser handles them.
	body := []byte(`{
		"model": "claude-opus-4-7",
		"messages": [{
			"role": "user",
			"content": [{
				"type": "image",
				"source": {"type": "url", "url": "https://example.com/x.png"}
			}]
		}]
	}`)
	got, err := extractContentFastAnthropic(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.FirstImageURL != "" {
		t.Fatalf("expected empty image url for non-base64 source, got %q", got.FirstImageURL)
	}
}

func TestExtractContentFastAnthropic_ToolUseCountedOnAssistant(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-7",
		"messages": [
			{"role": "user", "content": "search the docs"},
			{"role": "assistant", "content": [
				{"type": "text", "text": "let me search"},
				{"type": "tool_use", "id": "tu_1", "name": "web_search", "input": {"q": "x"}},
				{"type": "tool_use", "id": "tu_2", "name": "calc", "input": {}}
			]}
		]
	}`)
	got, err := extractContentFastAnthropic(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.AssistantToolCallCount != 2 {
		t.Fatalf("assistant tool calls: got %d, want 2", got.AssistantToolCallCount)
	}
	if len(got.AssistantToolNames) != 2 || got.AssistantToolNames[0] != "web_search" || got.AssistantToolNames[1] != "calc" {
		t.Fatalf("assistant tool names: got %+v", got.AssistantToolNames)
	}
}

func TestExtractContentFastAnthropic_ToolResultCountedOnUser(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-7",
		"messages": [
			{"role": "user", "content": "search"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "tu_1", "name": "web_search", "input": {}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "tu_1", "content": "result"}
			]}
		]
	}`)
	got, err := extractContentFastAnthropic(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ToolMessageCount != 1 || got.ToolResultCount != 1 {
		t.Fatalf("tool message/result counts: got %d/%d, want 1/1", got.ToolMessageCount, got.ToolResultCount)
	}
	// Both user turns still counted; the tool_result-bearing user turn does
	// not have plain text content so it should not become UserContent.
	if got.UserMessageCount != 2 {
		t.Fatalf("user message count: got %d, want 2", got.UserMessageCount)
	}
	if got.UserContent != "search" {
		t.Fatalf("user content: got %q, want %q", got.UserContent, "search")
	}
}

func TestExtractContentFastAnthropic_CountsToolDefinitions(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-7",
		"messages": [{"role": "user", "content": "x"}],
		"tools": [
			{"name": "a", "input_schema": {}},
			{"name": "b", "input_schema": {}},
			{"name": "c", "input_schema": {}}
		]
	}`)
	got, err := extractContentFastAnthropic(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ToolDefinitionCount != 3 {
		t.Fatalf("tool definitions: got %d, want 3", got.ToolDefinitionCount)
	}
}

func TestExtractRequestSignalsForProtocol_DispatchesByProtocol(t *testing.T) {
	openaiBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}`)
	anthropicBody := []byte(`{"model":"claude-opus-4-7","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)

	// OpenAI default — uses the original walker; string content extracted.
	got, err := extractRequestSignalsForProtocol("", openaiBody)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.UserContent != "hi" || got.Model != "gpt-4" {
		t.Fatalf("openai dispatch unexpected: %+v", got)
	}

	// Anthropic — the variant walker handles inline array text blocks.
	got, err = extractRequestSignalsForProtocol("anthropic", anthropicBody)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.UserContent != "hi" || got.Model != "claude-opus-4-7" {
		t.Fatalf("anthropic dispatch unexpected: %+v", got)
	}
}
