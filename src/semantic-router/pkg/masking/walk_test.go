package masking

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// scanForValue is a stub ScanFunc: it reports one span wherever value occurs
// in text, with the given entity type. No model is needed to exercise Apply.
func scanForValue(entityType, value string) ScanFunc {
	return func(text string) ([]Span, error) {
		idx := strings.Index(text, value)
		if idx < 0 {
			return nil, nil
		}
		return []Span{{EntityType: entityType, Start: idx, End: idx + len(value), Confidence: 1.0}}, nil
	}
}

// Case: System instruction.
func TestApply_SystemInstructionMasked(t *testing.T) {
	request := &llmprotocol.Request{
		Instructions: []llmprotocol.InstructionBlock{
			{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "contact alice@example.com for policy questions"},
			}},
		},
	}

	result, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "contact [EMAIL_ADDRESS_0] for policy questions"; request.Instructions[0].Content[0].Text != want {
		t.Fatalf("got %q, want %q", request.Instructions[0].Content[0].Text, want)
	}
	if !result.Changed {
		t.Fatalf("expected Changed=true")
	}
}

// Case: User message.
func TestApply_UserMessageMasked(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "email me at alice@example.com please"},
			}},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "email me at [EMAIL_ADDRESS_0] please"; request.Messages[0].Content[0].Text != want {
		t.Fatalf("got %q, want %q", request.Messages[0].Content[0].Text, want)
	}
}

// Case: Assistant message.
func TestApply_AssistantMessageMasked(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "sure, reach alice@example.com"},
			}},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "sure, reach [EMAIL_ADDRESS_0]"; request.Messages[0].Content[0].Text != want {
		t.Fatalf("got %q, want %q", request.Messages[0].Content[0].Text, want)
	}
}

// Case: Cross-protocol parity. The same conversation, decoded from three wire
// formats, must produce identical masked text once it reaches the shared
// neutral shape — this is the issue's first completion criterion, proven
// locally with a stub scanner.
func TestApply_CrossProtocolParity(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	formats := []llmprotocol.WireFormat{
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.AnthropicMessagesV1,
	}
	bodies := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1: `{"model":"m","messages":[` +
			`{"role":"user","content":"email me at alice@example.com please"}]}`,
		llmprotocol.OpenAIResponsesV1: `{"model":"m","input":[{"type":"message","role":"user",` +
			`"content":[{"type":"input_text","text":"email me at alice@example.com please"}]}]}`,
		llmprotocol.AnthropicMessagesV1: `{"model":"m","max_tokens":1024,"messages":[` +
			`{"role":"user","content":"email me at alice@example.com please"}]}`,
	}

	want := "email me at [EMAIL_ADDRESS_0] please"
	for _, format := range formats {
		request, _, _, err := engine.DecodeRequestForMutation(format, []byte(bodies[format]))
		if err != nil {
			t.Fatalf("decode %s: %v", format, err)
		}
		if _, err := Apply(&request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com")); err != nil {
			t.Fatalf("apply %s: %v", format, err)
		}
		if len(request.Messages) != 1 || len(request.Messages[0].Content) != 1 {
			t.Fatalf("%s: unexpected neutral shape: %+v", format, request.Messages)
		}
		if got := request.Messages[0].Content[0].Text; got != want {
			t.Fatalf("%s: got %q, want %q", format, got, want)
		}
	}
}

// Case: Tool result text.
func TestApply_ToolResultTextMasked(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
					CallID: "call_1",
					Content: []llmprotocol.Content{
						{Kind: llmprotocol.ContentText, Text: "lookup found alice@example.com"},
					},
				}},
			}},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	got := request.Messages[0].Content[0].ToolResult.Content[0].Text
	if want := "lookup found [EMAIL_ADDRESS_0]"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// Case: Tool call arguments — types, keys and structure survive.
func TestApply_ToolCallArgumentsMasked(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
					ID: "call_1", Name: "send_email",
					Arguments: `{"to":"a@x.com","n":3,"ok":true}`,
				}},
			}},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "a@x.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decoded map[string]any
	rawArgs := request.Messages[0].Content[0].ToolCall.Arguments
	if err := json.Unmarshal([]byte(rawArgs), &decoded); err != nil {
		t.Fatalf("masked arguments are not valid JSON: %v (%s)", err, rawArgs)
	}
	if decoded["to"] != "[EMAIL_ADDRESS_0]" {
		t.Fatalf("expected to=[EMAIL_ADDRESS_0], got %v", decoded["to"])
	}
	if n, ok := decoded["n"].(float64); !ok || n != 3 {
		t.Fatalf("expected n=3 (number), got %v (%T)", decoded["n"], decoded["n"])
	}
	if ok, isBool := decoded["ok"].(bool); !isBool || !ok {
		t.Fatalf("expected ok=true (bool), got %v (%T)", decoded["ok"], decoded["ok"])
	}
}

// Case: Nested JSON — masked in place, structure intact.
func TestApply_NestedJSONArgumentsMasked(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
					ID: "call_1", Name: "nested",
					Arguments: `{"a":{"b":["a@x.com"]}}`,
				}},
			}},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "a@x.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decoded map[string]any
	rawArgs := request.Messages[0].Content[0].ToolCall.Arguments
	if err := json.Unmarshal([]byte(rawArgs), &decoded); err != nil {
		t.Fatalf("masked arguments are not valid JSON: %v (%s)", err, rawArgs)
	}
	inner, ok := decoded["a"].(map[string]any)
	if !ok {
		t.Fatalf("expected a to remain an object, got %T", decoded["a"])
	}
	list, ok := inner["b"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("expected b to remain a one-element array, got %v", inner["b"])
	}
	if list[0] != "[EMAIL_ADDRESS_0]" {
		t.Fatalf("expected masked leaf, got %v", list[0])
	}
}

// Case: JSON keys — a key that looks like PII is never scanned or altered.
func TestApply_JSONKeysUnchanged(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
					ID: "call_1", Name: "keyed",
					Arguments: `{"alice@example.com":"contact alice@example.com"}`,
				}},
			}},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var decoded map[string]any
	rawArgs := request.Messages[0].Content[0].ToolCall.Arguments
	if err := json.Unmarshal([]byte(rawArgs), &decoded); err != nil {
		t.Fatalf("masked arguments are not valid JSON: %v (%s)", err, rawArgs)
	}
	value, ok := decoded["alice@example.com"]
	if !ok {
		t.Fatalf("expected key %q to survive unchanged, got keys %v", "alice@example.com", decoded)
	}
	if value != "contact [EMAIL_ADDRESS_0]" {
		t.Fatalf("expected value to be masked, got %v", value)
	}
}

// Case: Invalid JSON arguments abort the request.
func TestApply_InvalidJSONArgumentsErrors(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
					ID: "call_1", Name: "broken", Arguments: "{not json",
				}},
			}},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "a@x.com")); err == nil {
		t.Fatalf("expected an error for invalid JSON arguments")
	}
}

// Case: Reasoning block — never masked, signature intact.
func TestApply_ReasoningBlockUnchanged(t *testing.T) {
	original := "thinking about alice@example.com"
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentReasoning, Text: original, Signature: "sig-xyz"},
			}},
		},
	}

	result, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	block := request.Messages[0].Content[0]
	if block.Text != original {
		t.Fatalf("reasoning text was modified: got %q, want %q", block.Text, original)
	}
	if block.Signature != "sig-xyz" {
		t.Fatalf("reasoning signature was modified: got %q", block.Signature)
	}
	if result.Changed {
		t.Fatalf("expected Changed=false, reasoning must not be masked")
	}
}

// Case: Tools untouched — operator-authored tool definitions are never
// scanned, even when their description looks like PII.
func TestApply_ToolsUnchanged(t *testing.T) {
	request := &llmprotocol.Request{
		Tools: []llmprotocol.Tool{
			{Name: "lookup", Description: "contact alice@example.com for access"},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "contact alice@example.com for access"; request.Tools[0].Description != want {
		t.Fatalf("tool description was modified: got %q, want %q", request.Tools[0].Description, want)
	}
}

// Case: Media untouched — Data, URL, FileID and Filename are all left alone
// (requirement 5's full media list, including Filename which the table
// groups under "Media untouched").
func TestApply_MediaFieldsUnchanged(t *testing.T) {
	block := llmprotocol.Content{
		Kind:     llmprotocol.ContentImage,
		Data:     "base64-data-alice@example.com",
		URL:      "https://host/alice@example.com",
		FileID:   "file_alice@example.com",
		Filename: "alice@example.com.png",
	}
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{block}}},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	got := request.Messages[0].Content[0]
	if got.Data != block.Data || got.URL != block.URL || got.FileID != block.FileID || got.Filename != block.Filename {
		t.Fatalf("media fields were modified: got %+v, want %+v", got, block)
	}
}

// Case: Correlation intact — ToolCall.ID and ToolResult.CallID are never
// touched, even though they are string fields on masked content kinds.
func TestApply_CorrelationIdentifiersUnchanged(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
					ID: "call_abc", Name: "x", Arguments: `{"safe":"value"}`,
				}},
			}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
					CallID:  "call_abc",
					Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "no pii here"}},
				}},
			}},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "a@x.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if id := request.Messages[0].Content[0].ToolCall.ID; id != "call_abc" {
		t.Fatalf("ToolCall.ID was modified: got %q", id)
	}
	if callID := request.Messages[1].Content[0].ToolResult.CallID; callID != "call_abc" {
		t.Fatalf("ToolResult.CallID was modified: got %q", callID)
	}
}

// Case: Scan error aborts the walk, and the block being scanned when the
// error occurred is left unmodified.
func TestApply_ScanErrorAbortsAndLeavesRequestUnchanged(t *testing.T) {
	original := "trigger a scan failure"
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: original},
			}},
		},
	}
	failingScan := func(string) ([]Span, error) { return nil, fmt.Errorf("classifier unavailable") }

	result, err := Apply(request, NewAllocator(defaultCfg()), failingScan)
	if err == nil {
		t.Fatalf("expected an error")
	}
	if result.Changed || result.MaskedCount != 0 || result.CitationsDropped != 0 || len(result.EntityTypes) != 0 {
		t.Fatalf("expected zero-value Result on error, got %+v", result)
	}
	if got := request.Messages[0].Content[0].Text; got != original {
		t.Fatalf("request was mutated despite scan error: got %q, want %q", got, original)
	}
}

// Case: Empty text short-circuits without calling scan.
func TestApply_EmptyTextSkipsScan(t *testing.T) {
	called := false
	scan := func(string) ([]Span, error) {
		called = true
		return nil, nil
	}
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: ""}}},
		},
	}

	result, err := Apply(request, NewAllocator(defaultCfg()), scan)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if called {
		t.Fatalf("scan must not be called for empty text")
	}
	if result.Changed {
		t.Fatalf("expected Changed=false")
	}
}

// Case: Same value across blocks gets the same placeholder in both.
func TestApply_SameValueAcrossBlocksSharesPlaceholder(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "first mention: alice@example.com"},
			}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "second mention: alice@example.com"},
			}},
		},
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	first := request.Messages[0].Content[0].Text
	second := request.Messages[1].Content[0].Text
	if want := "first mention: [EMAIL_ADDRESS_0]"; first != want {
		t.Fatalf("got %q, want %q", first, want)
	}
	if want := "second mention: [EMAIL_ADDRESS_0]"; second != want {
		t.Fatalf("got %q, want %q", second, want)
	}
}

// Case: Result hygiene — the Result never carries a raw value or placeholder.
func TestApply_ResultHygiene(t *testing.T) {
	request := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "reach me at alice@example.com now"},
			}},
		},
	}

	result, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Changed || result.MaskedCount == 0 {
		t.Fatalf("expected the request to be masked, got %+v", result)
	}
	dump := fmt.Sprintf("%+v", result)
	if strings.Contains(dump, "alice@example.com") {
		t.Fatalf("Result leaked the raw value: %s", dump)
	}
	if strings.Contains(dump, "[EMAIL_ADDRESS_0]") {
		t.Fatalf("Result leaked the placeholder: %s", dump)
	}
}

// Requirement 5 also names Request.Model, Request.Metadata,
// PreviousResponseID and ConversationID — none of the 17 table rows exercise
// these directly, but the skip list is a security contract, so each gets an
// explicit assertion here.
func TestApply_RequestLevelFieldsUnchanged(t *testing.T) {
	request := &llmprotocol.Request{
		Model:              "model-alice@example.com",
		Metadata:           map[string]string{"alice@example.com": "note-alice@example.com"},
		PreviousResponseID: "resp_alice@example.com",
		ConversationID:     "conv_alice@example.com",
	}

	if _, err := Apply(request, NewAllocator(defaultCfg()), scanForValue("EMAIL_ADDRESS", "alice@example.com")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if request.Model != "model-alice@example.com" {
		t.Fatalf("Model was modified: got %q", request.Model)
	}
	if request.Metadata["alice@example.com"] != "note-alice@example.com" || len(request.Metadata) != 1 {
		t.Fatalf("Metadata was modified: got %v", request.Metadata)
	}
	if request.PreviousResponseID != "resp_alice@example.com" {
		t.Fatalf("PreviousResponseID was modified: got %q", request.PreviousResponseID)
	}
	if request.ConversationID != "conv_alice@example.com" {
		t.Fatalf("ConversationID was modified: got %q", request.ConversationID)
	}
}
