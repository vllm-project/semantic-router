package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A coding agent that reads an image file returns it in a tool result, and
// Chat Completions tool messages carry text only. Strict translation keeps
// refusing it. With tool_result.content accepted, the images follow the tool
// results as one user message, and an image-only result keeps a short text.
func TestChatToolResultMediaOptIn(t *testing.T) {
	body := []byte(`{"model":"m","max_tokens":64,"messages":[
		{"role":"user","content":"compare the plots"},
		{"role":"assistant","content":[
			{"type":"tool_use","id":"toolu_1","name":"Read","input":{"file_path":"a.png"}},
			{"type":"tool_use","id":"toolu_2","name":"Read","input":{"file_path":"b.png"}}]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"a.png"},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"YQ=="}}]},
			{"type":"tool_result","tool_use_id":"toolu_2","cache_control":{"type":"ephemeral"},"content":[
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"Yg=="}}]},
			{"type":"text","text":"which is smoother?"}]}]}`)
	route := func(request *llmprotocol.Request) error { request.Model = "routed"; return nil }
	strict, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	require.NoError(t, err)
	_, err = strict.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, route)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "tool_result_media")

	policy := llmprotocol.DefaultPolicy()
	policy.AllowedLossyFields = []string{llmprotocol.LossyFieldToolResultContent}
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	require.NoError(t, err)
	result, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, route)
	require.NoError(t, err)
	var wire struct {
		Messages []struct {
			Role       string          `json:"role"`
			ToolCallID string          `json:"tool_call_id"`
			Content    json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(result.Body, &wire))
	var roles []string
	for _, message := range wire.Messages {
		roles = append(roles, message.Role)
	}
	require.Equal(t, []string{"user", "assistant", "tool", "tool", "user", "user"}, roles, string(result.Body))
	require.Contains(t, string(wire.Messages[2].Content), "a.png")
	require.Contains(t, string(wire.Messages[3].Content), toolResultMovedText)
	require.Contains(t, string(wire.Messages[3].Content), "cache_control")
	require.Contains(t, string(wire.Messages[4].Content), "data:image/png;base64,YQ==")
	require.Contains(t, string(wire.Messages[4].Content), "data:image/png;base64,Yg==")
	require.Contains(t, string(wire.Messages[5].Content), "which is smoother?")
	var reported int
	for _, diagnostic := range result.Diagnostics {
		if diagnostic.Field == llmprotocol.LossyFieldToolResultContent {
			reported++
		}
	}
	require.Equal(t, 1, reported, "%+v", result.Diagnostics)
}

// Codex returns a viewed image as an input_image in its function call output,
// which reaches a Chat backend the same way.
func TestChatToolResultMediaOptInFromResponses(t *testing.T) {
	body := []byte(`{"model":"m","input":[
		{"type":"message","role":"user","content":[{"type":"input_text","text":"look"}]},
		{"type":"function_call","call_id":"call_1","name":"view_image","arguments":"{\"path\":\"a.png\"}"},
		{"type":"function_call_output","call_id":"call_1","output":[{"type":"input_image","image_url":"data:image/png;base64,YQ==","detail":"auto"}]}]}`)
	route := func(request *llmprotocol.Request) error { request.Model = "routed"; return nil }
	policy := llmprotocol.DefaultPolicy()
	policy.AllowedLossyFields = []string{llmprotocol.LossyFieldToolResultContent}
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	require.NoError(t, err)
	result, err := engine.TranslateRequest(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, body, route)
	require.NoError(t, err)
	var wire struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(result.Body, &wire))
	require.Len(t, wire.Messages, 4, string(result.Body))
	require.Equal(t, "tool", wire.Messages[2].Role)
	require.Equal(t, "user", wire.Messages[3].Role)
	require.Contains(t, string(wire.Messages[3].Content), "data:image/png;base64,YQ==")
}
