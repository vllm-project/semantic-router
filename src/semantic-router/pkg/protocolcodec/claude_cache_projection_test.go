package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const claudeCachedToolResult = `{
	"model":"m","max_tokens":64,
	"system":[{"type":"text","text":"cached instructions","cache_control":{"type":"ephemeral","ttl":"1h"}}],
	"messages":[
		{"role":"user","content":[{"type":"text","text":"lookup weather","cache_control":{"type":"ephemeral"}}]},
		{"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":"lookup","input":{"query":"weather"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","content":[
			{"type":"text","text":"sunny","cache_control":{"type":"ephemeral","ttl":"5m"}},
			{"type":"text","text":"72 F"}
		],"cache_control":{"type":"ephemeral","ttl":"1h"}}]}
	],
	"tools":[{"name":"lookup","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}]
}`

func TestClaudeToolResultOuterCacheProjectsToChatLastText(t *testing.T) {
	result, err := NewBuiltinEngine().TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1,
		[]byte(claudeCachedToolResult), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	var wire chatRequestWire
	if err := json.Unmarshal(result.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Messages) != 4 {
		t.Fatalf("Chat messages = %d, want four: %s", len(wire.Messages), result.Body)
	}
	toolResult := wire.Messages[3]
	if toolResult.Role != "tool" || toolResult.ToolCallID != "call-1" {
		t.Fatalf("tool result identity lost: %+v", toolResult)
	}
	var parts []chatContentWire
	if err := json.Unmarshal(toolResult.Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 || parts[0].Text != "sunny" || parts[1].Text != "72 F" ||
		parts[0].CacheControl == nil || parts[0].CacheControl.TTL != "5m" ||
		parts[1].CacheControl == nil || parts[1].CacheControl.TTL != "1h" {
		t.Fatalf("tool result cache boundary changed: %+v", parts)
	}
	for _, diagnostic := range result.Diagnostics {
		if diagnostic.Field == "cache_control" && diagnostic.Action == llmprotocol.DiagnosticDropped {
			t.Fatalf("Chat cache boundary was dropped: %+v", diagnostic)
		}
	}
}

func TestClaudeToolResultConflictingCacheBoundariesRejectChat(t *testing.T) {
	body := strings.Replace(claudeCachedToolResult,
		`"text":"72 F"`,
		`"text":"72 F","cache_control":{"type":"ephemeral","ttl":"5m"}`, 1)
	_, err := NewBuiltinEngine().TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, []byte(body), nil,
	)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_cache_directive" {
		t.Fatalf("conflicting cache boundaries returned %v, want unsupported_cache_directive", err)
	}
}

func TestClaudeCacheBoundariesOmitExplicitlyForResponses(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.AnthropicMessagesV1, []byte(claudeCachedToolResult))
	if err != nil {
		t.Fatal(err)
	}
	if !llmprotocol.RequiredCapabilities(request).Supports(llmprotocol.CapabilityCacheDirectives) {
		t.Fatal("Anthropic cache boundaries were not decoded")
	}
	result, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Diagnostics) != 1 || result.Diagnostics[0].Field != "cache_control" ||
		result.Diagnostics[0].Source != llmprotocol.AnthropicMessagesV1 ||
		result.Diagnostics[0].Target != llmprotocol.OpenAIResponsesV1 ||
		result.Diagnostics[0].Action != llmprotocol.DiagnosticDropped ||
		!strings.Contains(result.Diagnostics[0].Reason, "5 Anthropic prompt-cache boundaries") {
		t.Fatalf("cache omission was not explicit: %+v", result.Diagnostics)
	}
	if bytes.Contains(result.Body, []byte("cache_control")) ||
		!bytes.Contains(result.Body, []byte(`"type":"function_call_output"`)) ||
		!bytes.Contains(result.Body, []byte(`"call_id":"call-1"`)) ||
		!bytes.Contains(result.Body, []byte(`"name":"lookup"`)) {
		t.Fatalf("Responses lost tool-loop content or leaked unsupported cache controls: %s", result.Body)
	}
	if llmprotocol.RequiredCapabilities(result.Request).Supports(llmprotocol.CapabilityCacheDirectives) ||
		!llmprotocol.RequiredCapabilities(request).Supports(llmprotocol.CapabilityCacheDirectives) ||
		result.Request.Generation <= request.Generation {
		t.Fatalf("projection mutated source or kept stale envelope: original=%+v projected=%+v", request, result.Request)
	}
	translated, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, []byte(claudeCachedToolResult), nil)
	if err != nil || len(translated.Diagnostics) != 1 || translated.Diagnostics[0].Field != "cache_control" {
		t.Fatalf("translation did not report cache omission: err=%v diagnostics=%+v", err, translated.Diagnostics)
	}
}

func TestNonAnthropicCacheDirectiveStillRejectsResponses(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}]}`)
	_, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, body, nil)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_capability" ||
		!strings.Contains(protocolError.Message, "cache_directives") {
		t.Fatalf("unrelated cache directives unexpectedly passed: %v", err)
	}
}

func TestDirectResponsesCodecCannotSilentlyOmitCacheDirective(t *testing.T) {
	request, envelope, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, []byte(claudeCachedToolResult))
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = (OpenAIResponsesCodec{}).EncodeRequest(request, envelope, llmprotocol.DefaultPolicy())
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Code != "unsupported_cache_directive" {
		t.Fatalf("direct Responses codec silently omitted cache directives: %v", err)
	}
}
