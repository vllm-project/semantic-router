package anthropic

import (
	"strings"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
)

func TestParseAnthropicRequest_RejectsInvalidJSON(t *testing.T) {
	if _, _, err := ParseAnthropicRequest([]byte("{not json")); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestParseAnthropicRequest_RequiresModel(t *testing.T) {
	_, _, err := ParseAnthropicRequest([]byte(`{"messages":[{"role":"user","content":"hi"}]}`))
	if err == nil {
		t.Fatal("expected error when model missing")
	}
}

func TestParseAnthropicRequest_StampsSourceProtocol(t *testing.T) {
	_, ext, err := ParseAnthropicRequest([]byte(`{"model":"claude-opus-4-7","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ext == nil || ext.SourceProtocol != SourceProtocolAnthropic {
		t.Fatalf("expected SourceProtocol=anthropic, got %+v", ext)
	}
}

func TestParseAnthropicRequest_PlainTextRoundTrip(t *testing.T) {
	body := []byte(`{"model":"claude-opus-4-7","max_tokens":1024,"messages":[{"role":"user","content":"hello world"}]}`)
	params, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.Model != "claude-opus-4-7" {
		t.Fatalf("model: got %q", params.Model)
	}
	if params.MaxTokens.Value != 1024 {
		t.Fatalf("max_tokens: got %d, want 1024", params.MaxTokens.Value)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("messages: got %d, want 1", len(params.Messages))
	}
	if params.Messages[0].OfUser == nil {
		t.Fatalf("expected user message, got %+v", params.Messages[0])
	}
	if got := params.Messages[0].OfUser.Content.OfString.Value; got != "hello world" {
		t.Fatalf("user content: got %q, want hello world", got)
	}
	if ext.SourceProtocol != SourceProtocolAnthropic {
		t.Fatalf("source protocol: got %q", ext.SourceProtocol)
	}
}

func TestParseAnthropicRequest_DefaultsMaxTokens(t *testing.T) {
	body := []byte(`{"model":"claude","messages":[{"role":"user","content":"x"}]}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.MaxTokens.Value != DefaultMaxTokens {
		t.Fatalf("expected default max_tokens=%d, got %d", DefaultMaxTokens, params.MaxTokens.Value)
	}
}

func TestParseAnthropicRequest_SystemString(t *testing.T) {
	body := []byte(`{"model":"claude","system":"be helpful","messages":[{"role":"user","content":"hi"}]}`)
	params, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 2 {
		t.Fatalf("expected 2 messages (system+user), got %d", len(params.Messages))
	}
	if params.Messages[0].OfSystem == nil {
		t.Fatalf("expected system message at index 0, got %+v", params.Messages[0])
	}
	if len(ext.SystemBlocks) != 0 {
		t.Fatalf("expected no SystemBlocks for string system, got %+v", ext.SystemBlocks)
	}
}

func TestParseAnthropicRequest_SystemArrayWithCacheControl(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"system": [
			{"type": "text", "text": "block A", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
			{"type": "text", "text": "block B"}
		],
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	params, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 2 {
		t.Fatalf("expected system+user, got %d", len(params.Messages))
	}
	if len(ext.SystemBlocks) != 2 {
		t.Fatalf("expected 2 SystemBlocks, got %d", len(ext.SystemBlocks))
	}
	if got := ext.CacheControl["system.0"].TTL; got != "5m" {
		t.Fatalf("cache_control ttl for system.0: got %q, want 5m", got)
	}
	if _, present := ext.CacheControl["system.1"]; present {
		t.Fatal("expected no cache_control for system.1")
	}
}

func TestParseAnthropicRequest_UserArrayWithText(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "part one"},
				{"type": "text", "text": "part two"}
			]
		}]
	}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 {
		t.Fatalf("expected one user message, got %d", len(params.Messages))
	}
	parts := params.Messages[0].OfUser.Content.OfArrayOfContentParts
	if len(parts) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(parts))
	}
}

func TestParseAnthropicRequest_ImageBase64(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "look"},
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR=="}}
			]
		}]
	}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := params.Messages[0].OfUser.Content.OfArrayOfContentParts
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	if parts[1].OfImageURL == nil {
		t.Fatalf("expected image part at index 1, got %+v", parts[1])
	}
	if got := parts[1].OfImageURL.ImageURL.URL; got != "data:image/png;base64,iVBOR==" {
		t.Fatalf("image url: got %q", got)
	}
}

func TestParseAnthropicRequest_ImageURL(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/x.png"}}]}]
	}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := params.Messages[0].OfUser.Content.OfArrayOfContentParts
	if parts[0].OfImageURL == nil || parts[0].OfImageURL.ImageURL.URL != "https://example.com/x.png" {
		t.Fatalf("image url unexpected: %+v", parts[0])
	}
}

func TestParseAnthropicRequest_ImageFileID(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{"role":"user","content":[{"type":"image","source":{"type":"file","file_id":"file_abc"}}]}]
	}`)
	params, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parts := params.Messages[0].OfUser.Content.OfArrayOfContentParts
	if parts[0].OfImageURL == nil || parts[0].OfImageURL.ImageURL.URL != "file_id:file_abc" {
		t.Fatalf("image file_id url unexpected: %+v", parts[0])
	}
	if !hasWarning(ext, "image_file_id_unresolved") {
		t.Fatalf("expected image_file_id_unresolved warning, got %+v", ext.Warnings)
	}
}

func TestParseAnthropicRequest_ToolUseAndToolResult(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [
			{"role": "user", "content": "search the docs"},
			{"role": "assistant", "content": [
				{"type": "text", "text": "looking it up"},
				{"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "vsr"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "tu_1", "content": "vsr is a router"}
			]}
		]
	}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(params.Messages))
	}
	if params.Messages[1].OfAssistant == nil {
		t.Fatalf("expected assistant message at index 1")
	}
	if len(params.Messages[1].OfAssistant.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call on assistant, got %d", len(params.Messages[1].OfAssistant.ToolCalls))
	}
	call := params.Messages[1].OfAssistant.ToolCalls[0]
	if call.ID != "tu_1" || call.Function.Name != "search" {
		t.Fatalf("tool call unexpected: %+v", call)
	}
	if !strings.Contains(call.Function.Arguments, `"q"`) {
		t.Fatalf("tool arguments missing q: %q", call.Function.Arguments)
	}
	if params.Messages[2].OfTool == nil {
		t.Fatalf("expected tool message at index 2")
	}
	if params.Messages[2].OfTool.ToolCallID != "tu_1" {
		t.Fatalf("tool_call_id: got %q", params.Messages[2].OfTool.ToolCallID)
	}
	if params.Messages[2].OfTool.Content.OfString.Value != "vsr is a router" {
		t.Fatalf("tool content: got %q", params.Messages[2].OfTool.Content.OfString.Value)
	}
}

func TestParseAnthropicRequest_ToolResultArrayContent(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "user",
			"content": [{
				"type": "tool_result",
				"tool_use_id": "tu_1",
				"content": [
					{"type": "text", "text": "first chunk"},
					{"type": "text", "text": "second chunk"},
					{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}
				]
			}]
		}]
	}`)
	params, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Messages) != 1 || params.Messages[0].OfTool == nil {
		t.Fatalf("expected single tool message, got %+v", params.Messages)
	}
	gotContent := params.Messages[0].OfTool.Content.OfString.Value
	if !strings.Contains(gotContent, "first chunk") || !strings.Contains(gotContent, "second chunk") {
		t.Fatalf("expected both text chunks in content, got %q", gotContent)
	}
	if !hasWarning(ext, "tool_result_image_dropped") {
		t.Fatalf("expected tool_result_image_dropped warning, got %+v", ext.Warnings)
	}
}

func TestParseAnthropicRequest_ToolResultIsError(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "user",
			"content": [{
				"type": "tool_result",
				"tool_use_id": "tu_99",
				"is_error": true,
				"content": "boom"
			}]
		}]
	}`)
	_, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !hasWarningWithDetail(ext, "tool_result_is_error", "tu_99") {
		t.Fatalf("expected tool_result_is_error warning for tu_99, got %+v", ext.Warnings)
	}
}

func TestParseAnthropicRequest_TopK(t *testing.T) {
	body := []byte(`{"model":"claude","top_k":40,"messages":[{"role":"user","content":"hi"}]}`)
	_, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ext.TopK == nil || *ext.TopK != 40 {
		t.Fatalf("top_k: got %+v, want 40", ext.TopK)
	}
}

func TestParseAnthropicRequest_MetadataUserID(t *testing.T) {
	body := []byte(`{"model":"claude","metadata":{"user_id":"user-abc"},"messages":[{"role":"user","content":"hi"}]}`)
	params, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ext.MetadataUserID != "user-abc" {
		t.Fatalf("metadata user id: got %q", ext.MetadataUserID)
	}
	if params.User.Value != "user-abc" {
		t.Fatalf("params.User: got %q", params.User.Value)
	}
}

func TestParseAnthropicRequest_Thinking(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"thinking": {"type": "enabled", "budget_tokens": 8000, "display": "summarized"},
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	_, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ext.Thinking == nil {
		t.Fatal("expected Thinking to be set")
	}
	if ext.Thinking.Type != "enabled" || ext.Thinking.BudgetTokens != 8000 || ext.Thinking.Display != "summarized" {
		t.Fatalf("thinking unexpected: %+v", ext.Thinking)
	}
}

func TestParseAnthropicRequest_ToolsWithStrict(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"name": "search", "description": "search docs", "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}, "strict": true},
			{"name": "calc", "input_schema": {"type": "object"}}
		]
	}`)
	params, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(params.Tools))
	}
	if params.Tools[0].Function.Name != "search" {
		t.Fatalf("tools[0] name: got %q", params.Tools[0].Function.Name)
	}
	if params.Tools[0].Function.Description.Value != "search docs" {
		t.Fatalf("tools[0] description: got %q", params.Tools[0].Function.Description.Value)
	}
	if !ext.ToolStrict["search"] {
		t.Fatalf("expected strict=true for search, got %+v", ext.ToolStrict)
	}
	if ext.ToolStrict["calc"] {
		t.Fatalf("expected strict=false (or absent) for calc, got %+v", ext.ToolStrict)
	}
}

func TestParseAnthropicRequest_ToolChoiceAuto(t *testing.T) {
	body := []byte(`{"model":"claude","tool_choice":{"type":"auto"},"messages":[{"role":"user","content":"x"}]}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.ToolChoice.OfAuto.Value != "auto" {
		t.Fatalf("tool_choice: got %+v", params.ToolChoice)
	}
}

func TestParseAnthropicRequest_ToolChoiceAny(t *testing.T) {
	body := []byte(`{"model":"claude","tool_choice":{"type":"any"},"messages":[{"role":"user","content":"x"}]}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.ToolChoice.OfAuto.Value != "required" {
		t.Fatalf("any → required mapping failed: got %+v", params.ToolChoice)
	}
}

func TestParseAnthropicRequest_ToolChoiceNamed(t *testing.T) {
	body := []byte(`{"model":"claude","tool_choice":{"type":"tool","name":"search"},"messages":[{"role":"user","content":"x"}]}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.ToolChoice.OfChatCompletionNamedToolChoice == nil {
		t.Fatal("expected named tool choice")
	}
	if params.ToolChoice.OfChatCompletionNamedToolChoice.Function.Name != "search" {
		t.Fatalf("named tool: got %+v", params.ToolChoice.OfChatCompletionNamedToolChoice)
	}
}

func TestParseAnthropicRequest_ToolChoiceDisableParallel(t *testing.T) {
	body := []byte(`{"model":"claude","tool_choice":{"type":"auto","disable_parallel_tool_use":true},"messages":[{"role":"user","content":"x"}]}`)
	params, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ext.ToolChoiceDisableParallel {
		t.Fatal("expected ToolChoiceDisableParallel=true")
	}
	if params.ParallelToolCalls.Value {
		t.Fatal("expected ParallelToolCalls=false on OpenAI side")
	}
}

func TestParseAnthropicRequest_TemperatureTopPStop(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"temperature": 0.7,
		"top_p": 0.9,
		"stop_sequences": ["END", "STOP"],
		"messages": [{"role": "user", "content": "x"}]
	}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.Temperature.Value != 0.7 {
		t.Fatalf("temperature: got %v", params.Temperature.Value)
	}
	if params.TopP.Value != 0.9 {
		t.Fatalf("top_p: got %v", params.TopP.Value)
	}
	if len(params.Stop.OfStringArray) != 2 {
		t.Fatalf("stop sequences: got %+v", params.Stop)
	}
}

func TestParseAnthropicRequest_DocumentBlock(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "user",
			"content": [{
				"type": "document",
				"source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="},
				"citations": {"enabled": true}
			}]
		}]
	}`)
	_, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ext.Documents) != 1 {
		t.Fatalf("expected 1 document, got %d", len(ext.Documents))
	}
	doc := ext.Documents[0]
	if doc.SourceType != "base64" || doc.MediaType != "application/pdf" {
		t.Fatalf("document fields unexpected: %+v", doc)
	}
	if !doc.Citations || !ext.CitationsEnabled {
		t.Fatal("expected citations enabled")
	}
}

func TestParseAnthropicRequest_ServerToolUse(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "assistant",
			"content": [{"type": "server_tool_use", "name": "web_search", "input": {"q": "x"}}]
		}]
	}`)
	_, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ext.ServerTools) != 1 {
		t.Fatalf("expected 1 server tool capture, got %d", len(ext.ServerTools))
	}
	if ext.ServerTools[0].Name != "web_search" {
		t.Fatalf("server tool name: got %q", ext.ServerTools[0].Name)
	}
}

func TestParseAnthropicRequest_ServerToolDefinition(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{"role": "user", "content": "x"}],
		"tools": [{"type": "web_search_20250305", "name": "web_search"}]
	}`)
	params, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(params.Tools) != 0 {
		t.Fatalf("expected no OpenAI tools (server tool captured instead), got %d", len(params.Tools))
	}
	if len(ext.ServerTools) != 1 || ext.ServerTools[0].Type != "web_search_20250305" {
		t.Fatalf("server tool capture unexpected: %+v", ext.ServerTools)
	}
}

func TestParseAnthropicRequest_RedactedThinkingWarnAndDrop(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "assistant",
			"content": [{"type": "redacted_thinking", "data": "opaque"}]
		}]
	}`)
	_, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !hasWarning(ext, "unsupported_block_type") {
		t.Fatalf("expected unsupported_block_type warning for redacted_thinking, got %+v", ext.Warnings)
	}
}

func TestParseAnthropicRequest_ThinkingBlockSignature(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "assistant",
			"content": [{"type": "thinking", "thinking": "let me think", "signature": "sig-xyz"}]
		}]
	}`)
	_, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := ext.ThinkingSignatures["messages[0].content[0]"]; got != "sig-xyz" {
		t.Fatalf("thinking signature: got %q, want sig-xyz", got)
	}
}

func TestParseAnthropicRequest_ToolUseInputPreservesStructuredJSON(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "assistant",
			"content": [{"type": "tool_use", "id": "tu_x", "name": "f", "input": {"nested": {"a": 1, "b": [1,2,3]}}}]
		}]
	}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	args := params.Messages[0].OfAssistant.ToolCalls[0].Function.Arguments
	if !strings.Contains(args, `"nested"`) || !strings.Contains(args, `[1,2,3]`) {
		t.Fatalf("structured input not preserved: %q", args)
	}
}

func TestParseAnthropicRequest_AnthropicBetaCaptured(t *testing.T) {
	body := []byte(`{"model":"claude","anthropic_beta":"files-api-2025-04-14","messages":[{"role":"user","content":"x"}]}`)
	_, ext, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ext.AnthropicBeta != "files-api-2025-04-14" {
		t.Fatalf("anthropic_beta: got %q", ext.AnthropicBeta)
	}
}

func TestParseAnthropicRequest_PreservesDocumentOrderUserToolUser(t *testing.T) {
	body := []byte(`{
		"model": "claude",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "before"},
				{"type": "tool_result", "tool_use_id": "tu_1", "content": "mid"},
				{"type": "text", "text": "after"}
			]
		}]
	}`)
	params, _, err := ParseAnthropicRequest(body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Expected message sequence: user(before), tool(mid), user(after).
	if len(params.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(params.Messages))
	}
	if params.Messages[0].OfUser == nil {
		t.Fatalf("msg[0] should be user")
	}
	// First user message uses array content of one text part.
	parts := params.Messages[0].OfUser.Content.OfArrayOfContentParts
	if len(parts) != 1 || parts[0].OfText.Text != "before" {
		t.Fatalf("first user content unexpected: %+v", parts)
	}
	if params.Messages[1].OfTool == nil {
		t.Fatalf("msg[1] should be tool")
	}
	if params.Messages[2].OfUser == nil {
		t.Fatalf("msg[2] should be user")
	}
}

// --- test helpers ---

func hasWarning(ext *ir.IRExtensions, reason ir.WarningReason) bool {
	if ext == nil {
		return false
	}
	for _, w := range ext.Warnings {
		if w.Reason == reason {
			return true
		}
	}
	return false
}

func hasWarningWithDetail(ext *ir.IRExtensions, reason ir.WarningReason, detail string) bool {
	if ext == nil {
		return false
	}
	for _, w := range ext.Warnings {
		if w.Reason == reason && w.Detail == detail {
			return true
		}
	}
	return false
}

// Compile-time guard: ensure ParseAnthropicRequest is callable in the
// shape downstream extproc dispatch expects.
var _ = func() (*openai.ChatCompletionNewParams, *ir.IRExtensions, error) {
	return ParseAnthropicRequest(nil)
}
