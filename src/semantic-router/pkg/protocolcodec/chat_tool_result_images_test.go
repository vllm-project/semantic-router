package protocolcodec

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const toolResultPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

type chatToolImageMessage struct {
	Role       string          `json:"role"`
	ToolCallID string          `json:"tool_call_id"`
	Content    json.RawMessage `json:"content"`
}

func chatMessagesOf(t *testing.T, body []byte) []chatToolImageMessage {
	t.Helper()
	var wire struct {
		Messages []chatToolImageMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("Chat body: %v", err)
	}
	return wire.Messages
}

func chatParts(t *testing.T, content json.RawMessage) []chatContentWire {
	t.Helper()
	var text string
	if json.Unmarshal(content, &text) == nil {
		return []chatContentWire{{Type: "text", Text: text}}
	}
	var parts []chatContentWire
	if err := json.Unmarshal(content, &parts); err != nil {
		t.Fatalf("content: %v", err)
	}
	return parts
}

func hasToolImageDiagnostic(diagnostics llmprotocol.Diagnostics) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "messages[].content.tool_result.image" && diagnostic.Action == llmprotocol.DiagnosticApproximated {
			return true
		}
	}
	return false
}

// A Chat tool message carries text only, so the images Claude Code's tool
// results hold (a screenshot it reads) follow the run of tool messages in one
// user message, and the tool messages answering the turn stay unbroken.
func TestClaudeToolResultImagesFollowTheToolMessagesForChat(t *testing.T) {
	body := `{
		"model":"m","max_tokens":64,
		"messages":[
			{"role":"user","content":"Compare the two screenshots."},
			{"role":"assistant","content":[
				{"type":"tool_use","id":"call-1","name":"Read","input":{"file_path":"a.png"}},
				{"type":"tool_use","id":"call-2","name":"Read","input":{"file_path":"b.png"}}]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"call-1","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + toolResultPNG + `"}}]},
				{"type":"tool_result","tool_use_id":"call-2","content":[{"type":"text","text":"b.png:"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + toolResultPNG + `"}}]}]}
		],
		"tools":[{"name":"Read","input_schema":{"type":"object"}}]
	}`
	result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(body), nil)
	if err != nil {
		t.Fatalf("a tool result holding an image was rejected: %v", err)
	}
	messages := chatMessagesOf(t, result.Body)
	roles := make([]string, len(messages))
	for i, message := range messages {
		roles[i] = message.Role
	}
	if len(messages) != 5 || roles[2] != "tool" || roles[3] != "tool" || roles[4] != "user" {
		t.Fatalf("roles = %v, want user, assistant, tool, tool, user", roles)
	}
	if first := chatParts(t, messages[2].Content); messages[2].ToolCallID != "call-1" || len(first) != 1 || first[0].Text != chatToolResultImagesFollow {
		t.Fatalf("an image-only result's tool message = %s", messages[2].Content)
	}
	if second := chatParts(t, messages[3].Content); messages[3].ToolCallID != "call-2" || len(second) != 1 || second[0].Text != "b.png:" {
		t.Fatalf("the tool message keeps its text alone: %s", messages[3].Content)
	}
	images := chatParts(t, messages[4].Content)
	if len(images) != 3 || images[0].Text != chatToolResultImagesNote || images[1].Type != "image_url" || images[2].Type != "image_url" ||
		images[1].ImageURL.URL != "data:image/png;base64,"+toolResultPNG {
		t.Fatalf("the user message carries the note and both images in order: %s", messages[4].Content)
	}
	if !hasToolImageDiagnostic(result.Diagnostics) {
		t.Fatalf("the move is reported: %+v", result.Diagnostics)
	}
}

// Codex's view_image answers a function call with an image; Chat gets the
// same tool message and user message.
func TestResponsesFunctionOutputImageFollowsForChat(t *testing.T) {
	body := `{
		"model":"m",
		"input":[
			{"role":"user","content":[{"type":"input_text","text":"What does shot.png show?"}]},
			{"type":"function_call","call_id":"call-1","name":"view_image","arguments":"{\"path\":\"shot.png\"}"},
			{"type":"function_call_output","call_id":"call-1","output":[{"type":"input_image","image_url":"data:image/png;base64,` + toolResultPNG + `"}]}
		],
		"tools":[{"type":"function","name":"view_image","parameters":{"type":"object"}}]
	}`
	result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, []byte(body), nil)
	if err != nil {
		t.Fatalf("a function output holding an image was rejected: %v", err)
	}
	messages := chatMessagesOf(t, result.Body)
	last := messages[len(messages)-1]
	tool := messages[len(messages)-2]
	if tool.Role != "tool" || tool.ToolCallID != "call-1" || last.Role != "user" {
		t.Fatalf("messages = %+v", messages)
	}
	if parts := chatParts(t, last.Content); len(parts) != 2 || parts[1].Type != "image_url" {
		t.Fatalf("the image follows in a user message: %s", last.Content)
	}
}

// An image-only result's cache boundary moves to its tool message's text.
func TestAnImageOnlyToolResultKeepsItsCacheBoundaryOnTheToolMessage(t *testing.T) {
	body := `{
		"model":"m","max_tokens":64,
		"messages":[
			{"role":"user","content":"Read it."},
			{"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":"Read","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","cache_control":{"type":"ephemeral"},
				"content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + toolResultPNG + `"}}]}]}
		],
		"tools":[{"name":"Read","input_schema":{"type":"object"}}]
	}`
	result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(body), nil)
	if err != nil {
		t.Fatalf("rejected: %v", err)
	}
	messages := chatMessagesOf(t, result.Body)
	parts := chatParts(t, messages[2].Content)
	if len(parts) != 1 || parts[0].Text != chatToolResultImagesFollow || parts[0].CacheControl == nil {
		t.Fatalf("the tool message holds the boundary: %s", messages[2].Content)
	}
}

// Other media a tool result holds still have no place in Chat.
func TestAToolResultDocumentStillHasNoPlaceInChat(t *testing.T) {
	body := `{
		"model":"m","max_tokens":64,
		"messages":[
			{"role":"user","content":"Read it."},
			{"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":"Read","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1",
				"content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}]}]}
		],
		"tools":[{"name":"Read","input_schema":{"type":"object"}}]
	}`
	_, err := NewBuiltinEngine().TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(body), nil)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "tool_result_media" {
		t.Fatalf("a tool result document = %v, want tool_result_media", err)
	}
}
