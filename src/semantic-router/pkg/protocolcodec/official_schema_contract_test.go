package protocolcodec

import (
	"encoding/json"
	"errors"
	"reflect"
	"sort"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOfficialNestedUnsupportedFieldsFailWithTypedErrors(t *testing.T) {
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{"Chat message name", llmprotocol.OpenAIChatV1, `{"model":"m","messages":[{"role":"user","name":"alice","content":"hello"}]}`},
		{"Chat cache breakpoint", llmprotocol.OpenAIChatV1, `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hello","prompt_cache_breakpoint":{"mode":"explicit"}}]}]}`},
		{"Responses deferred tool", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":"hello","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},"defer_loading":true}]}`},
		{"Responses reasoning mode", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":"hello","reasoning":{"mode":"pro"}}`},
		{"Responses reasoning summary", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":"hello","reasoning":{"summary":"concise"}}`},
		{"Responses reasoning context", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":"hello","reasoning":{"context":"all_turns"}}`},
		{"Responses deprecated reasoning summary", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":"hello","reasoning":{"generate_summary":"auto"}}`},
		{"Responses text verbosity", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":"hello","text":{"verbosity":"high"}}`},
		{"Responses input breakpoint", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello","prompt_cache_breakpoint":{"mode":"explicit"}}]}]}`},
		{"Responses input message phase", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":[{"type":"message","role":"user","phase":"commentary","content":[{"type":"input_text","text":"hello"}]}]}`},
		{"Anthropic eager tool", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"lookup","input_schema":{"type":"object"},"eager_input_streaming":true}]}`},
		{"Anthropic image transformation", llmprotocol.AnthropicMessagesV1, `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/image.png"},"transformations":{"on_load":{"type":"auto"}}}]}]}`},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(test.format, []byte(test.body))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature {
				t.Fatalf("nested field returned %T %v, want typed unsupported_feature", err, err)
			}
		})
	}
}

func TestAnthropicCacheDirectivesSurviveSemanticMutation(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","max_tokens":16,"system":[{"type":"text","text":"cached instructions","cache_control":{"type":"ephemeral","ttl":"1h"}}],"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"lookup","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}]}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	assertAnthropicCacheRequest(t, request)
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip struct {
		System []struct {
			CacheControl *anthropicCacheControlWire `json:"cache_control"`
		} `json:"system"`
		Tools []anthropicToolWire `json:"tools"`
	}
	if err := json.Unmarshal(encoded.Body, &roundTrip); err != nil {
		t.Fatal(err)
	}
	assertAnthropicCacheRoundTrip(t, roundTrip.System, roundTrip.Tools, encoded.Body)
}

func assertAnthropicCacheRequest(t *testing.T, request llmprotocol.Request) {
	t.Helper()
	if len(request.Instructions) != 1 || len(request.Instructions[0].Content) != 1 || len(request.Tools) != 1 {
		t.Fatalf("cache directive request shape = %+v", request)
	}
	cache, toolCache := request.Instructions[0].Content[0].Cache, request.Tools[0].Cache
	if cache == nil || cache.TTL != "1h" || toolCache == nil || toolCache.Type != "ephemeral" {
		t.Fatalf("cache directives were not decoded semantically: %+v", request)
	}
}

func assertAnthropicCacheRoundTrip(
	t *testing.T,
	system []struct {
		CacheControl *anthropicCacheControlWire `json:"cache_control"`
	},
	tools []anthropicToolWire,
	body []byte,
) {
	t.Helper()
	if len(system) != 1 || len(tools) != 1 {
		t.Fatalf("cache directive round-trip shape changed: %s", body)
	}
	if system[0].CacheControl == nil || system[0].CacheControl.TTL != "1h" || tools[0].CacheControl == nil {
		t.Fatalf("cache directives were not re-encoded after routing mutation: %s", body)
	}
}

func TestCacheDirectivesSurviveChatAndAnthropicTranslation(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{
		"model":"m",
		"messages":[
			{"role":"system","content":[
				{"type":"text","text":"stable preface"},
				{"type":"text","text":"cached instructions","cache_control":{"type":"ephemeral","ttl":"1h"}}
			]},
			{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral","ttl":"5m"}}]},
			{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call-1","content":[{"type":"text","text":"sunny","cache_control":{"type":"ephemeral"}}]}
		],
		"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}},"cache_control":{"type":"ephemeral"}}]
	}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	assertDecodedChatCacheDirectives(t, request)

	request.Model = "routed-model"
	request.Generation++
	chatRoundTrip, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	assertChatCacheRoundTrip(t, chatRoundTrip.Body)

	translated, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, body, func(request *llmprotocol.Request) error {
		request.Model = "anthropic-model"
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	assertTranslatedAnthropicCacheDirectives(t, translated.Body)
}

func assertDecodedChatCacheDirectives(t *testing.T, request llmprotocol.Request) {
	t.Helper()
	if len(request.Instructions) != 1 || len(request.Instructions[0].Content) != 2 ||
		len(request.Messages) != 3 || len(request.Tools) != 1 {
		t.Fatalf("Chat cache request shape = %+v", request)
	}
	if request.Instructions[0].Content[1].Cache == nil || request.Instructions[0].Content[1].Cache.TTL != "1h" ||
		request.Messages[0].Content[0].Cache == nil || request.Tools[0].Cache == nil ||
		request.Messages[2].Content[0].ToolResult.Content[0].Cache == nil {
		t.Fatalf("Chat cache directives were not decoded semantically: %+v", request)
	}
}

func assertChatCacheRoundTrip(t *testing.T, body []byte) {
	t.Helper()
	var wire chatRequestWire
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Messages) != 4 || len(wire.Tools) != 1 {
		t.Fatalf("Chat cache round-trip shape changed: %s", body)
	}
	system, user, result := decodeChatContentFixture(t, wire.Messages[0].Content),
		decodeChatContentFixture(t, wire.Messages[1].Content), decodeChatContentFixture(t, wire.Messages[3].Content)
	if len(system) != 2 || len(user) != 1 || len(result) != 1 {
		t.Fatalf("Chat cache content shape changed: %s", body)
	}
	assertChatCacheWireValues(t, system, user, result, wire.Tools, body)
}

func assertChatCacheWireValues(
	t *testing.T,
	system, user, result []chatContentWire,
	tools []chatToolWire,
	body []byte,
) {
	t.Helper()
	if system[0].Text != "stable preface" || system[0].CacheControl != nil ||
		system[1].CacheControl == nil || system[1].CacheControl.TTL != "1h" ||
		user[0].CacheControl == nil || result[0].CacheControl == nil || tools[0].CacheControl == nil {
		t.Fatalf("Chat cache directives changed after semantic routing mutation: %s", body)
	}
}

func decodeChatContentFixture(t *testing.T, body json.RawMessage) []chatContentWire {
	t.Helper()
	var content []chatContentWire
	if err := json.Unmarshal(body, &content); err != nil {
		t.Fatal(err)
	}
	return content
}

func assertTranslatedAnthropicCacheDirectives(t *testing.T, body []byte) {
	t.Helper()
	var wire anthropicRequestWire
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Messages) != 3 {
		t.Fatalf("Anthropic cache translation shape changed: %s", body)
	}
	system := decodeAnthropicContentFixture(t, wire.System)
	user := decodeAnthropicContentFixture(t, wire.Messages[0].Content)
	result := decodeAnthropicContentFixture(t, wire.Messages[2].Content)
	tools := decodeAnthropicToolsFixture(t, wire.Tools)
	if len(system) != 2 || len(user) != 1 || len(result) != 1 || len(tools) != 1 {
		t.Fatalf("Anthropic cache translation content shape changed: %s", body)
	}
	nested := decodeAnthropicContentFixture(t, result[0].Content)
	assertAnthropicCacheWireValues(t, system, user, result, nested, tools, body)
}

func assertAnthropicCacheWireValues(
	t *testing.T,
	system, user, result, nested []anthropicContentWire,
	tools []anthropicToolWire,
	body []byte,
) {
	t.Helper()
	if system[0].Text != "stable preface" || system[0].CacheControl != nil ||
		system[1].CacheControl == nil || system[1].CacheControl.TTL != "1h" ||
		user[0].CacheControl == nil || result[0].Type != "tool_result" ||
		len(nested) != 1 || nested[0].CacheControl == nil || tools[0].CacheControl == nil {
		t.Fatalf("cache directives did not retain order and ownership after Chat to Anthropic translation: %s", body)
	}
}

func decodeAnthropicContentFixture(t *testing.T, body json.RawMessage) []anthropicContentWire {
	t.Helper()
	var content []anthropicContentWire
	if err := json.Unmarshal(body, &content); err != nil {
		t.Fatal(err)
	}
	return content
}

func decodeAnthropicToolsFixture(t *testing.T, body json.RawMessage) []anthropicToolWire {
	t.Helper()
	var tools []anthropicToolWire
	if err := json.Unmarshal(body, &tools); err != nil {
		t.Fatal(err)
	}
	return tools
}

func TestCacheDirectiveOnUnsupportedChatBlockFailsExplicitly(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":"lookup","input":{},"cache_control":{"type":"ephemeral"}}]}]}`)
	_, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature {
		t.Fatalf("unsupported Chat cache directive returned %T %v, want typed unsupported_feature", err, err)
	}
}

func TestMalformedChatCacheDirectiveFailsValidation(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"forever","ttl":"24h"}}]}]}`)
	_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorInvalidRequest || protocolError.Code != "invalid_cache_directive" {
		t.Fatalf("malformed Chat cache directive returned %T %v, want invalid_cache_directive", err, err)
	}
}

func TestAnthropicThinkingDisableAndParallelToolPolicySurviveSemanticMutation(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","max_tokens":16,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}],"tool_choice":{"type":"tool","name":"lookup","disable_parallel_tool_use":true}}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	assertDisabledAnthropicThinkingPolicy(t, request)
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip anthropicRequestWire
	if err := json.Unmarshal(encoded.Body, &roundTrip); err != nil {
		t.Fatal(err)
	}
	assertDisabledAnthropicThinkingWire(t, &roundTrip, encoded.Body)
	_, err = engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, body, nil)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature {
		t.Fatalf("reasoning disable cross-protocol translation returned %T %v, want typed unsupported_feature", err, err)
	}
}

func assertDisabledAnthropicThinkingPolicy(t *testing.T, request llmprotocol.Request) {
	t.Helper()
	if request.ReasoningMode != llmprotocol.ReasoningModeDisabled ||
		request.ParallelToolCalls == nil || *request.ParallelToolCalls {
		t.Fatalf("Anthropic request policy was not decoded semantically: %+v", request)
	}
}

func assertDisabledAnthropicThinkingWire(t *testing.T, wire *anthropicRequestWire, body []byte) {
	t.Helper()
	if wire.Thinking == nil || wire.Thinking.Type != "disabled" || wire.ToolChoice == nil {
		t.Fatalf("Anthropic request policy changed after routing mutation: %s", body)
	}
	if wire.ToolChoice.DisableParallelToolUse == nil || !*wire.ToolChoice.DisableParallelToolUse {
		t.Fatalf("Anthropic parallel-tool policy changed after routing mutation: %s", body)
	}
}

func TestOfficialAnthropicAdaptiveThinkingSurvivesSemanticMutation(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","max_tokens":16,"thinking":{"type":"adaptive"},"messages":[{"role":"user","content":"hello"}]}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if request.ReasoningMode != llmprotocol.ReasoningModeAdaptive {
		t.Fatalf("reasoning mode = %q", request.ReasoningMode)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip anthropicRequestWire
	if err := json.Unmarshal(encoded.Body, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if roundTrip.Thinking == nil || roundTrip.Thinking.Type != "adaptive" {
		t.Fatalf("adaptive thinking changed after routing mutation: %s", encoded.Body)
	}
	_, err = engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature {
		t.Fatalf("adaptive thinking cross-protocol translation returned %T %v, want typed unsupported_feature", err, err)
	}
}

func TestAnthropicThinkingSignatureSurvivesResponseMutationOrFailsExplicitly(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{
		"id":"message_1","type":"message","role":"assistant","model":"claude-model",
		"content":[
			{"type":"thinking","thinking":"private reasoning","signature":"signed-block"},
			{"type":"text","text":"final answer"}
		],
		"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":3}
	}`)
	response, envelope, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	assertDecodedAnthropicThinkingSignature(t, response)
	response.Model = "public-model"
	response.Generation++
	encoded, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, response, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire anthropicResponseWire
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	var content []anthropicContentWire
	if err := json.Unmarshal(wire.Content, &content); err != nil {
		t.Fatal(err)
	}
	assertEncodedAnthropicThinkingSignature(t, content, encoded.Body)

	_, err = engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, body, nil)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature {
		t.Fatalf("cross-protocol thinking signature returned %T %v, want typed unsupported_feature", err, err)
	}
}

func assertDecodedAnthropicThinkingSignature(t *testing.T, response llmprotocol.Response) {
	t.Helper()
	if len(response.Output) != 1 || len(response.Output[0].Content) != 2 {
		t.Fatalf("Anthropic thinking response shape = %+v", response.Output)
	}
	thinking := response.Output[0].Content[0]
	if thinking.Kind != llmprotocol.ContentReasoning || thinking.Signature != "signed-block" {
		t.Fatalf("Anthropic thinking signature was not decoded: %+v", response.Output)
	}
}

func assertEncodedAnthropicThinkingSignature(t *testing.T, content []anthropicContentWire, body []byte) {
	t.Helper()
	if len(content) != 2 {
		t.Fatalf("Anthropic thinking content shape changed: %s", body)
	}
	if content[0].Type != "thinking" || content[0].Thinking != "private reasoning" ||
		content[0].Signature != "signed-block" {
		t.Fatalf("Anthropic thinking signature changed after response mutation: %s", body)
	}
}

func TestAnthropicBufferedResponsePreservesOrderedThinkingTextAndToolBlocks(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{
		"id":"message_1","type":"message","role":"assistant","model":"claude-model",
		"content":[
			{"type":"thinking","thinking":"first thought","signature":"signature-1"},
			{"type":"text","text":"checking"},
			{"type":"thinking","thinking":"second thought","signature":"signature-2"},
			{"type":"tool_use","id":"call-1","name":"lookup","input":{"q":"weather"}}
		],
		"stop_reason":"tool_use","usage":{"input_tokens":4,"output_tokens":7}
	}`)
	response, envelope, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	response.Model = "public-model"
	response.Generation++
	encoded, err := engine.EncodeResponse(llmprotocol.AnthropicMessagesV1, response, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire anthropicResponseWire
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	var content []anthropicContentWire
	if err := json.Unmarshal(wire.Content, &content); err != nil {
		t.Fatal(err)
	}
	assertOrderedAnthropicContent(t, content, encoded.Body)
}

func assertOrderedAnthropicContent(t *testing.T, content []anthropicContentWire, body []byte) {
	t.Helper()
	if len(content) != 4 {
		t.Fatalf("ordered Anthropic content shape changed: %s", body)
	}
	first, text, second, tool := content[0], content[1], content[2], content[3]
	assertAnthropicThinkingBlock(t, first, "first thought", "signature-1", body)
	assertAnthropicTextBlock(t, text, body)
	assertAnthropicThinkingBlock(t, second, "second thought", "signature-2", body)
	assertAnthropicToolBlock(t, tool, body)
}

func assertAnthropicThinkingBlock(
	t *testing.T,
	content anthropicContentWire,
	thinking, signature string,
	body []byte,
) {
	t.Helper()
	if content.Type != "thinking" || content.Thinking != thinking || content.Signature != signature {
		t.Fatalf("thinking block changed: %s", body)
	}
}

func assertAnthropicTextBlock(t *testing.T, content anthropicContentWire, body []byte) {
	t.Helper()
	if content.Type != "text" || content.Text != "checking" {
		t.Fatalf("text block changed: %s", body)
	}
}

func assertAnthropicToolBlock(t *testing.T, content anthropicContentWire, body []byte) {
	t.Helper()
	if content.Type != "tool_use" || content.ID != "call-1" || content.Name != "lookup" ||
		!jsonSemanticallyEqual(content.Input, []byte(`{"q":"weather"}`)) {
		t.Fatalf("tool block changed: %s", body)
	}
}

func TestOfficialUnsupportedResponsesItemDiscriminatorsAreTyped(t *testing.T) {
	supported := fields("function_call", "function_call_output", "item_reference", "message", "reasoning")
	unsupported := fields(
		"additional_tools", "apply_patch_call", "apply_patch_call_output", "code_interpreter_call",
		"compaction", "compaction_trigger", "computer_call", "computer_call_output", "custom_tool_call",
		"custom_tool_call_output", "file_search_call", "function_shell_call",
		"function_shell_call_output", "image_generation_call", "local_shell_call", "local_shell_call_output",
		"mcp_approval_request", "mcp_approval_response", "mcp_call", "mcp_list_tools", "program",
		"program_output", "tool_search_call", "tool_search_output", "web_search_call",
	)
	assertClosedDiscriminatorInventory(t, "OpenAI Responses input item", 30, supported, unsupported)
	engine := NewBuiltinEngine()
	for _, itemType := range unsupported {
		t.Run(itemType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"model": "m",
				"input": []map[string]any{{
					"type": itemType, "id": "item_1", "variant_specific_field": true,
				}},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature {
				t.Fatalf("item %q returned %T %v, want typed unsupported_feature", itemType, err, err)
			}
		})
	}
}

func TestOfficialUnsupportedResponsesOutputItemDiscriminatorsAreTyped(t *testing.T) {
	unsupported := fields(
		"additional_tools", "apply_patch_call", "apply_patch_call_output", "code_interpreter_call",
		"compaction", "computer_call", "computer_call_output", "custom_tool_call",
		"custom_tool_call_output", "file_search_call", "function_call_output", "function_shell_call",
		"function_shell_call_output", "image_generation_call", "local_shell_call", "local_shell_call_output",
		"mcp_approval_request", "mcp_approval_response", "mcp_call", "mcp_list_tools", "program",
		"program_output", "tool_search_call", "tool_search_output", "web_search_call",
	)
	assertClosedDiscriminatorInventory(
		t,
		"OpenAI Responses output item",
		28,
		fields("function_call", "message", "reasoning"),
		unsupported,
	)
	engine := NewBuiltinEngine()
	for _, itemType := range unsupported {
		t.Run(itemType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"id": "resp_1", "model": "m", "status": "completed",
				"output": []map[string]any{{
					"type": itemType, "id": "item_1", "variant_specific_field": true,
				}},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_output_item")
		})
	}
}

func TestOfficialResponsesItemVariantsRejectCrossVariantFields(t *testing.T) {
	engine := NewBuiltinEngine()
	requestCases := []string{
		`{"model":"m","input":[{"type":"message","role":"user","content":"hello","arguments":"{}"}]}`,
		`{"model":"m","input":[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","content":[]}]}`,
		`{"model":"m","input":[{"type":"reasoning","id":"reason_1","summary":[],"output":"wrong"}]}`,
	}
	for index, body := range requestCases {
		t.Run("request/"+string(rune('a'+index)), func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(body))
			assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_input_item_variant")
		})
	}

	responseCases := []string{
		`{"id":"resp_1","model":"m","status":"completed","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[],"arguments":"{}"}]}`,
		`{"id":"resp_1","model":"m","status":"completed","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{}","content":[]}]}`,
		`{"id":"resp_1","model":"m","status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[],"content":[],"output":"wrong"}]}`,
	}
	for index, body := range responseCases {
		t.Run("response/"+string(rune('a'+index)), func(t *testing.T) {
			_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, []byte(body))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_response_item")
		})
	}
}

func TestOfficialUnsupportedResponsesToolDiscriminatorsAreTyped(t *testing.T) {
	unsupported := fields(
		"apply_patch", "code_interpreter", "computer", "computer_use_preview", "custom",
		"file_search", "image_generation", "local_shell", "mcp", "namespace",
		"programmatic_tool_calling", "shell", "tool_search", "web_search", "web_search_preview",
	)
	assertClosedDiscriminatorInventory(t, "OpenAI Responses tool", 16, fields("function"), unsupported)
	engine := NewBuiltinEngine()
	for _, toolType := range unsupported {
		t.Run(toolType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"model": "m",
				"input": "hello",
				"tools": []map[string]any{{"type": toolType, "variant_specific_field": true}},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature || protocolError.Code != "unsupported_tool" {
				t.Fatalf("tool %q returned %T %v, want unsupported_feature/unsupported_tool", toolType, err, err)
			}
		})
	}
}

func TestOfficialResponsesContentDiscriminatorInventoriesAreClosed(t *testing.T) {
	assertClosedDiscriminatorInventory(
		t,
		"OpenAI Responses input content",
		3,
		fields("input_file", "input_image", "input_text"),
		nil,
	)
	assertClosedDiscriminatorInventory(
		t,
		"OpenAI Responses output message content",
		2,
		fields("output_text", "refusal"),
		nil,
	)
	assertClosedDiscriminatorInventory(
		t,
		"OpenAI Responses reasoning content",
		1,
		fields("reasoning_text"),
		nil,
	)
}

func TestOfficialResponsesRequestContentUnionsArePositionScoped(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name     string
		body     string
		category llmprotocol.ErrorCategory
		code     string
	}{
		{
			name:     "instructions are string only",
			body:     `{"model":"m","instructions":[{"type":"input_text","text":"system"}],"input":"hello"}`,
			category: llmprotocol.ErrorInvalidRequest,
			code:     "invalid_instructions",
		},
		{
			name:     "user message rejects output content",
			body:     `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"output_text","text":"hello"}]}]}`,
			category: llmprotocol.ErrorUnsupportedFeature,
			code:     "unsupported_content",
		},
		{
			name:     "function output rejects provider output content",
			body:     `{"model":"m","input":[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"call_1","output":[{"type":"output_text","text":"done"}]}]}`,
			category: llmprotocol.ErrorUnsupportedFeature,
			code:     "unsupported_content",
		},
		{
			name:     "assistant history rejects mixed schema unions",
			body:     `{"model":"m","input":[{"type":"message","role":"assistant","content":[{"type":"input_text","text":"first"},{"type":"output_text","text":"second"}]}]}`,
			category: llmprotocol.ErrorInvalidRequest,
			code:     "mixed_assistant_content",
		},
		{
			name:     "generic text block is not a Responses input alias",
			body:     `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"text","text":"hello"}]}]}`,
			category: llmprotocol.ErrorUnsupportedFeature,
			code:     "unsupported_content",
		},
		{
			name:     "tool role must use function output item",
			body:     `{"model":"m","input":[{"type":"message","role":"tool","content":"done"}]}`,
			category: llmprotocol.ErrorInvalidRequest,
			code:     "invalid_role",
		},
		{
			name:     "input text cannot carry image fields",
			body:     `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello","image_url":"https://example.com/image.png"}]}]}`,
			category: llmprotocol.ErrorInvalidRequest,
			code:     "invalid_content_variant",
		},
		{
			name:     "input image cannot carry file fields",
			body:     `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","detail":"auto","image_url":"https://example.com/image.png","filename":"wrong.txt"}]}]}`,
			category: llmprotocol.ErrorInvalidRequest,
			code:     "invalid_content_variant",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, []byte(test.body))
			assertProtocolError(t, err, test.category, test.code)
		})
	}

	for _, contentType := range fields("input_text", "output_text") {
		t.Run("assistant history accepts official "+contentType+" union", func(t *testing.T) {
			body := []byte(`{"model":"m","input":[{"type":"message","role":"assistant","content":[{"type":"` + contentType + `","text":"history"}]},{"type":"message","role":"user","content":"continue"}]}`)
			if _, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body); err != nil {
				t.Fatalf("official assistant history was rejected: %v", err)
			}
		})
	}
}

func TestOfficialResponsesFileDetailRoundTrips(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"file_1","detail":"high"}]}]}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 1 || len(request.Messages[0].Content) != 1 || request.Messages[0].Content[0].Detail != "high" {
		t.Fatalf("file detail was not decoded: %+v", request.Messages)
	}
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]any
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	input := wire["input"].([]any)
	message := input[0].(map[string]any)
	content := message["content"].([]any)
	if got := content[0].(map[string]any)["detail"]; got != "high" {
		t.Fatalf("file detail round-trip = %v, want high", got)
	}
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		if _, err := engine.TranslateRequest(llmprotocol.OpenAIResponsesV1, target, body, nil); err == nil {
			t.Fatalf("%s silently dropped Responses file detail", target)
		}
	}
}

func TestOfficialResponsesProviderOutputContentIsStrict(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name string
		body string
		code string
	}{
		{
			name: "request-only input text",
			body: `{"id":"resp_1","model":"m","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"input_text","text":"bad"}]}]}`,
			code: "invalid_response_content",
		},
		{
			name: "reasoning text in output message",
			body: `{"id":"resp_1","model":"m","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"reasoning_text","text":"bad"}]}]}`,
			code: "invalid_response_content",
		},
		{
			name: "string shorthand",
			body: `{"id":"resp_1","model":"m","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":"bad"}]}`,
			code: "invalid_response_content",
		},
		{
			name: "non-assistant output role",
			body: `{"id":"resp_1","model":"m","status":"completed","output":[{"id":"msg_1","type":"message","role":"user","status":"completed","content":[{"type":"output_text","text":"bad","annotations":[],"logprobs":[]}]}]}`,
			code: "invalid_response_role",
		},
		{
			name: "output text cannot carry input fields",
			body: `{"id":"resp_1","model":"m","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"bad","annotations":[],"logprobs":[],"image_url":"https://example.com/image.png"}]}]}`,
			code: "invalid_response_content",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, []byte(test.body))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

func TestOfficialResponsesReasoningContentIsPreserved(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","model":"m","status":"completed",
		"output":[{
			"id":"rs_1","type":"reasoning","status":"completed",
			"summary":[{"type":"summary_text","text":"short summary"}],
			"content":[{"type":"reasoning_text","text":"full reasoning"}]
		}]
	}`)
	engine := NewBuiltinEngine()
	response, envelope, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	assertResponsesReasoningContent(t, response)
	response.Generation++
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, response, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire responsesResponseWire
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	var items []responsesItemWire
	if err := json.Unmarshal(wire.Output, &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || len(items[0].Summary) == 0 || len(items[0].Content) == 0 {
		t.Fatalf("reasoning summary/content scopes changed on encode: %s", encoded.Body)
	}
}

func assertResponsesReasoningContent(t *testing.T, response llmprotocol.Response) {
	t.Helper()
	if len(response.Output) != 1 || len(response.Output[0].Content) != 2 {
		t.Fatalf("reasoning response shape = %+v", response.Output)
	}
	summary, content := response.Output[0].Content[0], response.Output[0].Content[1]
	if summary.Kind != llmprotocol.ContentReasoning || summary.Text != "short summary" ||
		summary.Reasoning != llmprotocol.ReasoningScopeSummary {
		t.Fatalf("reasoning summary was not preserved: %+v", response.Output)
	}
	if content.Kind != llmprotocol.ContentReasoning || content.Text != "full reasoning" ||
		content.Reasoning != llmprotocol.ReasoningScopeText {
		t.Fatalf("reasoning content was not preserved: %+v", response.Output)
	}
}

func TestOfficialUnsupportedAnthropicToolDiscriminatorsAreTyped(t *testing.T) {
	unsupported := fields(
		"bash_20250124", "browser_toolset_20260801",
		"code_execution_20250522", "code_execution_20250825", "code_execution_20260120", "code_execution_20260521",
		"computer_toolset_20260801", "memory_20250818",
		"text_editor_20250124", "text_editor_20250429", "text_editor_20250728",
		"tool_search_tool_bm25_20251119", "tool_search_tool_regex_20251119",
		"web_fetch_20250910", "web_fetch_20260209", "web_fetch_20260309", "web_fetch_20260318",
		"web_search_20250305", "web_search_20260209", "web_search_20260318",
	)
	assertClosedDiscriminatorInventory(t, "Anthropic tool", 21, fields("custom"), unsupported)
	engine := NewBuiltinEngine()
	for _, toolType := range unsupported {
		t.Run(toolType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"model":      "m",
				"max_tokens": 16,
				"messages":   []map[string]any{{"role": "user", "content": "hello"}},
				"tools": []map[string]any{{
					"type":                   toolType,
					"name":                   "server_tool",
					"input_schema":           map[string]any{"type": "object"},
					"variant_specific_field": true,
				}},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature || protocolError.Code != "unsupported_tool" {
				t.Fatalf("tool %q returned %T %v, want unsupported_feature/unsupported_tool", toolType, err, err)
			}
		})
	}
}

func TestOfficialUnsupportedChatCustomToolCallsAreTyped(t *testing.T) {
	engine := NewBuiltinEngine()
	request := []byte(`{
		"model":"m",
		"messages":[
			{"role":"user","content":"use the grammar"},
			{"role":"assistant","tool_calls":[{"id":"call_1","type":"custom","custom":{"name":"grammar","input":"answer"}}]}
		]
	}`)
	_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, request)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_tool_call")

	response := []byte(`{
		"id":"chatcmpl_1","object":"chat.completion","model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"custom","custom":{"name":"grammar","input":"answer"}}]},"finish_reason":"tool_calls"}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
	}`)
	_, _, _, err = engine.DecodeResponse(llmprotocol.OpenAIChatV1, response)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_tool_call")
}

func TestOfficialChatRequestContentBlockUnionsAreRoleScoped(t *testing.T) {
	all := fields("file", "image_url", "input_audio", "refusal", "text")
	roles := []struct {
		role          string
		officialCount int
		allowed       []string
	}{
		{role: "developer", officialCount: 1, allowed: fields("text")},
		{role: "system", officialCount: 1, allowed: fields("text")},
		{role: "user", officialCount: 4, allowed: fields("file", "image_url", "input_audio", "text")},
		{role: "assistant", officialCount: 2, allowed: fields("refusal", "text")},
		{role: "tool", officialCount: 1, allowed: fields("text")},
	}
	engine := NewBuiltinEngine()
	for _, role := range roles {
		assertClosedDiscriminatorInventory(t, "Chat Completions "+role.role+" content block", role.officialCount, role.allowed, nil)
		allowed := make(map[string]struct{}, len(role.allowed))
		for _, contentType := range role.allowed {
			allowed[contentType] = struct{}{}
		}
		for _, contentType := range all {
			if _, ok := allowed[contentType]; ok {
				continue
			}
			t.Run(role.role+"_rejects_"+contentType, func(t *testing.T) {
				message := map[string]any{
					"role":    role.role,
					"content": []map[string]any{{"type": contentType}},
				}
				if role.role == "tool" {
					message["tool_call_id"] = "call_1"
				}
				body, err := json.Marshal(map[string]any{"model": "m", "messages": []any{message}})
				if err != nil {
					t.Fatal(err)
				}
				_, _, _, err = engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
				assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_content")
			})
		}
	}

	for _, contentType := range fields("input_text", "output_text") {
		t.Run("rejects_non_chat_"+contentType, func(t *testing.T) {
			body := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"` + contentType + `","text":"hello"}]}]}`)
			_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_content")
		})
	}
}

func TestOfficialChatContentVariantsRejectCrossVariantFields(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name string
		body string
	}{
		{
			name: "text with image payload",
			body: `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hello","image_url":{"url":"https://example.com/image.png"}}]}]}`,
		},
		{
			name: "image with file payload",
			body: `{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"},"file":{"file_id":"file_1"}}]}]}`,
		},
		{
			name: "refusal with cache directive",
			body: `{"model":"m","messages":[{"role":"assistant","content":[{"type":"refusal","refusal":"no","cache_control":{"type":"ephemeral"}}]}]}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(test.body))
			assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_content_variant")
		})
	}
}

func TestOfficialChatResponseMessageShapeIsStrict(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name string
		body string
		code string
	}{
		{
			name: "content array",
			body: `{"id":"chatcmpl_1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":[{"type":"text","text":"hello"}]},"finish_reason":"stop"}]}`,
			code: "invalid_response_content",
		},
		{
			name: "non-assistant role",
			body: `{"id":"chatcmpl_1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"user","content":"hello"},"finish_reason":"stop"}]}`,
			code: "invalid_response_role",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, []byte(test.body))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

func TestOfficialUnsupportedToolChoiceDiscriminatorsAreTyped(t *testing.T) {
	engine := NewBuiltinEngine()
	chatUnsupported := fields("allowed_tools", "custom")
	assertClosedDiscriminatorInventory(
		t,
		"OpenAI Chat Completions object tool choice",
		3,
		fields("function"),
		chatUnsupported,
	)
	for _, choiceType := range chatUnsupported {
		t.Run("chat/"+choiceType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"model":       "m",
				"messages":    []map[string]any{{"role": "user", "content": "hello"}},
				"tool_choice": map[string]any{"type": choiceType},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_tool_choice")
		})
	}
	t.Run("chat/function rejects custom payload", func(t *testing.T) {
		body := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}],"tool_choice":{"type":"function","function":{"name":"lookup"},"custom":{"name":"wrong"}}}`)
		_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
		assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_tool_choice")
	})

	responsesUnsupported := fields(
		"allowed_tools", "apply_patch", "code_interpreter", "computer", "computer_use",
		"computer_use_preview", "custom", "file_search", "image_generation", "mcp",
		"programmatic_tool_calling", "shell", "web_search_preview", "web_search_preview_2025_03_11",
	)
	assertClosedDiscriminatorInventory(
		t,
		"OpenAI Responses object tool choice",
		15,
		fields("function"),
		responsesUnsupported,
	)
	for _, choiceType := range responsesUnsupported {
		t.Run("responses/"+choiceType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"model": "m", "input": "hello",
				"tool_choice": map[string]any{"type": choiceType},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_tool_choice")
		})
	}
	t.Run("responses/function rejects MCP payload", func(t *testing.T) {
		body := []byte(`{"model":"m","input":"hello","tool_choice":{"type":"function","name":"lookup","server_label":"wrong"}}`)
		_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
		assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_tool_choice")
	})
}

func TestOfficialAnthropicToolChoiceUnionIsClosedAndPositionScoped(t *testing.T) {
	assertClosedDiscriminatorInventory(
		t,
		"Anthropic tool choice",
		4,
		fields("any", "auto", "none", "tool"),
		nil,
	)
	engine := NewBuiltinEngine()
	tests := []struct {
		name   string
		choice string
		code   string
	}{
		{
			name:   "auto rejects tool name",
			choice: `{"type":"auto","name":"lookup"}`,
			code:   "invalid_tool_choice_variant",
		},
		{
			name:   "none rejects parallel control",
			choice: `{"type":"none","disable_parallel_tool_use":true}`,
			code:   "invalid_tool_choice_variant",
		},
		{
			name:   "named tool requires name",
			choice: `{"type":"tool"}`,
			code:   "invalid_tool_choice",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := []byte(`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hello"}],"tool_choice":` + test.choice + `}`)
			_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
			assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, test.code)
		})
	}
}

func TestOfficialAnthropicContentBlockUnionsAreClosed(t *testing.T) {
	requestSupported := fields("document", "image", "text", "thinking", "tool_result", "tool_use")
	requestUnsupported := fields(
		"bash_code_execution_tool_result", "code_execution_tool_result", "container_upload",
		"redacted_thinking", "search_result", "server_tool_use",
		"text_editor_code_execution_tool_result", "tool_search_tool_result",
		"web_fetch_tool_result", "web_search_tool_result",
	)
	responseSupported := fields("text", "thinking", "tool_use")
	responseUnsupported := fields(
		"bash_code_execution_tool_result", "code_execution_tool_result", "container_upload",
		"redacted_thinking", "server_tool_use", "text_editor_code_execution_tool_result",
		"tool_search_tool_result", "web_fetch_tool_result", "web_search_tool_result",
	)
	assertClosedDiscriminatorInventory(t, "Anthropic request content block", 16, requestSupported, requestUnsupported)
	assertClosedDiscriminatorInventory(t, "Anthropic response content block", 12, responseSupported, responseUnsupported)
	engine := NewBuiltinEngine()
	for _, blockType := range requestUnsupported {
		t.Run("request/"+blockType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"model": "m", "max_tokens": 16,
				"messages": []map[string]any{{
					"role": "user", "content": []map[string]any{{"type": blockType}},
				}},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "")
		})
	}
	for _, blockType := range responseUnsupported {
		t.Run("response/"+blockType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
				"content":     []map[string]any{{"type": blockType}},
				"stop_reason": "end_turn",
				"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "")
		})
	}
	for _, blockType := range fields("document", "image", "tool_result") {
		t.Run("response_rejects_request_only/"+blockType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"id": "msg_1", "type": "message", "role": "assistant", "model": "m",
				"content":     []map[string]any{{"type": blockType}},
				"stop_reason": "end_turn",
				"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, _, _, err = engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_content")
		})
	}
}

func TestOfficialAnthropicSystemMessagePreservesItsPositionAcrossTheMatrix(t *testing.T) {
	body := []byte(`{
		"model":"m","max_tokens":32,
		"messages":[
			{"role":"user","content":"before"},
			{"role":"system","content":"policy update"},
			{"role":"user","content":"after"}
		]
	}`)
	engine := NewBuiltinEngine()
	for _, target := range builtinFormats {
		t.Run(string(target), func(t *testing.T) {
			translated, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, target, body, nil)
			if err != nil {
				t.Fatal(err)
			}
			roles := requestWireMessageRoles(t, target, translated.Body)
			if !reflect.DeepEqual(roles, []string{"user", "system", "user"}) {
				t.Fatalf("system message position changed: roles=%v body=%s", roles, translated.Body)
			}
		})
	}
}

func requestWireMessageRoles(t *testing.T, format llmprotocol.WireFormat, body []byte) []string {
	t.Helper()
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	field := "messages"
	if format == llmprotocol.OpenAIResponsesV1 {
		field = "input"
	}
	var messages []struct {
		Type string `json:"type"`
		Role string `json:"role"`
	}
	if err := json.Unmarshal(wire[field], &messages); err != nil {
		t.Fatal(err)
	}
	roles := make([]string, 0, len(messages))
	for _, message := range messages {
		if format != llmprotocol.OpenAIResponsesV1 || message.Type == "message" {
			roles = append(roles, message.Role)
		}
	}
	return roles
}

func TestOfficialAnthropicResponseRejectsRequestContentShorthand(t *testing.T) {
	body := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"m",
		"content":"not an official response content array","stop_reason":"end_turn",
		"usage":{"input_tokens":1,"output_tokens":1}
	}`)
	_, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_response_content")
}

func TestOfficialAnthropicContentVariantsRejectCrossVariantFields(t *testing.T) {
	engine := NewBuiltinEngine()
	request := []byte(`{
		"model":"m","max_tokens":16,
		"messages":[{"role":"user","content":[{"type":"text","text":"hello","source":{"type":"url","url":"https://example.com/image.png"}}]}]
	}`)
	_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, request)
	assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_content_variant")

	response := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"m",
		"content":[{"type":"thinking","thinking":"work","signature":"sig","text":"wrong variant"}],
		"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}
	}`)
	_, _, _, err = engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, response)
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "invalid_response_content")
}

func TestOfficialAnthropicMediaSourceUnionIsClosed(t *testing.T) {
	engine := NewBuiltinEngine()
	supported := []struct {
		name       string
		blockType  string
		sourceType string
		source     map[string]any
	}{
		{"image/base64", "image", "base64", map[string]any{"media_type": "image/png", "data": "aW1hZ2U="}},
		{"image/url", "image", "url", map[string]any{"url": "https://example.com/image.png"}},
		{"image/file", "image", "file", map[string]any{"file_id": "file_image"}},
		{"document/base64", "document", "base64", map[string]any{"media_type": "application/pdf", "data": "cGRm"}},
		{"document/url", "document", "url", map[string]any{"url": "https://example.com/document.pdf"}},
		{"document/file", "document", "file", map[string]any{"file_id": "file_document"}},
	}
	for _, test := range supported {
		t.Run("supported/"+test.name, func(t *testing.T) {
			assertSupportedAnthropicMediaSource(t, engine, test.blockType, test.sourceType, test.source)
		})
	}

	unsupported := []struct {
		name   string
		source map[string]any
	}{
		{"document/text", map[string]any{"type": "text", "media_type": "text/plain", "data": "plain text"}},
		{"document/content", map[string]any{"type": "content", "content": []map[string]any{{"type": "text", "text": "document"}}}},
	}
	for _, test := range unsupported {
		t.Run("unsupported/"+test.name, func(t *testing.T) {
			assertAnthropicMediaSourceError(
				t, engine, "document", test.source,
				llmprotocol.ErrorUnsupportedFeature, "unsupported_document_source",
			)
		})
	}

	invalid := []struct {
		name      string
		blockType string
		source    map[string]any
	}{
		{"missing discriminator", "image", map[string]any{"url": "https://example.com/image.png"}},
		{"unknown discriminator", "image", map[string]any{"type": "future", "url": "https://example.com/image.png"}},
		{"mixed variants", "image", map[string]any{"type": "url", "url": "https://example.com/image.png", "file_id": "file_1"}},
		{"document source used for image", "image", map[string]any{"type": "text", "media_type": "text/plain", "data": "text"}},
	}
	for _, test := range invalid {
		t.Run("invalid/"+test.name, func(t *testing.T) {
			assertAnthropicMediaSourceError(
				t, engine, test.blockType, test.source,
				llmprotocol.ErrorInvalidRequest, "invalid_media_source",
			)
		})
	}
}

func assertSupportedAnthropicMediaSource(
	t *testing.T,
	engine *Engine,
	blockType, sourceType string,
	fields map[string]any,
) {
	t.Helper()
	source := map[string]any{"type": sourceType}
	for key, value := range fields {
		source[key] = value
	}
	body, err := anthropicMediaRequest(blockType, source)
	if err != nil {
		t.Fatal(err)
	}
	request, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 1 || len(request.Messages[0].Content) != 1 {
		t.Fatalf("decoded media shape = %+v", request.Messages)
	}
}

func assertAnthropicMediaSourceError(
	t *testing.T,
	engine *Engine,
	blockType string,
	source map[string]any,
	category llmprotocol.ErrorCategory,
	code string,
) {
	t.Helper()
	body, err := anthropicMediaRequest(blockType, source)
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	assertProtocolError(t, err, category, code)
}

func anthropicMediaRequest(blockType string, source map[string]any) ([]byte, error) {
	return json.Marshal(map[string]any{
		"model": "m", "max_tokens": 16,
		"messages": []map[string]any{{
			"role":    "user",
			"content": []map[string]any{{"type": blockType, "source": source}},
		}},
	})
}

func TestOfficialAdditionalToolsCacheBreakpointIsExplicitlyUnsupported(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","input":[{"type":"additional_tools","id":"item_1","role":"assistant","tools":[],"prompt_cache_breakpoint":{"mode":"explicit"}}]}`)
	_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature {
		t.Fatalf("additional_tools cache breakpoint returned %T %v, want typed unsupported_feature", err, err)
	}
}

func TestOfficialFileInputsTranslateAcrossTheOpenAIMatrix(t *testing.T) {
	engine := NewBuiltinEngine()
	assertOfficialChatFileInput(t, engine)
	assertOfficialResponsesFileInput(t, engine)
}

func assertOfficialChatFileInput(t *testing.T, engine *Engine) {
	t.Helper()
	chatBody := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"brief.txt","file_data":"data:text/plain;base64,aGVsbG8="}}]}]}`)
	request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, chatBody)
	if err != nil {
		t.Fatal(err)
	}
	file := request.Messages[0].Content[0]
	if file.Kind != llmprotocol.ContentFile || file.Filename != "brief.txt" || file.MediaType != "text/plain" || file.Data != "aGVsbG8=" {
		t.Fatalf("Chat file semantics = %+v", file)
	}
	translated, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, chatBody, nil)
	if err != nil {
		t.Fatal(err)
	}
	var output map[string]json.RawMessage
	if err := json.Unmarshal(translated.Body, &output); err != nil || len(output["input"]) == 0 {
		t.Fatalf("Chat to Responses file translation = %s (%v)", translated.Body, err)
	}
}

func assertOfficialResponsesFileInput(t *testing.T, engine *Engine) {
	t.Helper()
	responsesBody := []byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_file","file_url":"https://example.com/brief.pdf","filename":"brief.pdf"}]}]}`)
	request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, responsesBody)
	if err != nil {
		t.Fatal(err)
	}
	file := request.Messages[0].Content[0]
	if file.Kind != llmprotocol.ContentFile || file.URL != "https://example.com/brief.pdf" || file.Filename != "brief.pdf" {
		t.Fatalf("Responses file URL semantics = %+v", file)
	}
}

func TestOfficialUnsupportedRequestFieldsFailWithTypedErrors(t *testing.T) {
	tests := []struct {
		format llmprotocol.WireFormat
		base   map[string]any
		fields []string
	}{
		{
			format: llmprotocol.OpenAIChatV1,
			base:   map[string]any{"model": "m", "messages": []any{map[string]any{"role": "user", "content": "hello"}}},
			fields: fields(
				"audio", "function_call", "functions", "logit_bias", "logprobs", "modalities", "moderation",
				"prediction", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention",
				"safety_identifier", "service_tier", "top_logprobs", "verbosity", "web_search_options",
			),
		},
		{
			format: llmprotocol.OpenAIResponsesV1,
			base:   map[string]any{"model": "m", "input": "hello"},
			fields: fields(
				"background", "context_management", "include", "max_tool_calls", "moderation", "prompt",
				"prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "safety_identifier",
				"service_tier", "top_logprobs",
			),
		},
		{
			format: llmprotocol.AnthropicMessagesV1,
			base: map[string]any{
				"model": "m", "max_tokens": 16,
				"messages": []any{map[string]any{"role": "user", "content": "hello"}},
			},
			fields: fields("cache_control", "container", "inference_geo", "service_tier"),
		},
	}

	engine := NewBuiltinEngine()
	for _, test := range tests {
		for _, field := range test.fields {
			t.Run(string(test.format)+"/"+field, func(t *testing.T) {
				body := cloneJSONMap(test.base)
				body[field] = true
				encoded, err := json.Marshal(body)
				if err != nil {
					t.Fatal(err)
				}
				_, _, _, err = engine.DecodeRequest(test.format, encoded)
				var protocolError *llmprotocol.ProtocolError
				if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorUnsupportedFeature {
					t.Fatalf("field %q returned %T %v, want typed unsupported_feature", field, err, err)
				}
			})
		}
	}
}

func TestOfficialNullableUnsupportedRequestFieldsAreTreatedAsAbsent(t *testing.T) {
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{
			name:   "Chat optional audio",
			format: llmprotocol.OpenAIChatV1,
			body:   `{"model":"m","messages":[{"role":"user","content":"hello"}],"audio":null}`,
		},
		{
			name:   "Responses optional background",
			format: llmprotocol.OpenAIResponsesV1,
			body:   `{"model":"m","input":"hello","background":null}`,
		},
		{
			name:   "Anthropic optional container",
			format: llmprotocol.AnthropicMessagesV1,
			body:   `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hello"}],"container":null}`,
		},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, _, _, err := engine.DecodeRequest(test.format, []byte(test.body)); err != nil {
				t.Fatalf("nullable unsupported field was not treated as absent: %v", err)
			}
		})
	}
}

func TestOfficialStreamOptionsAreExplicitAndPortable(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name            string
		format          llmprotocol.WireFormat
		body            string
		wantUsage       *bool
		wantObfuscation *bool
	}{
		{
			name:      "Chat explicit false",
			format:    llmprotocol.OpenAIChatV1,
			body:      `{"model":"m","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":false,"include_obfuscation":false}}`,
			wantUsage: boolPointer(false), wantObfuscation: boolPointer(false),
		},
		{
			name:      "Chat explicit true",
			format:    llmprotocol.OpenAIChatV1,
			body:      `{"model":"m","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true,"include_obfuscation":true}}`,
			wantUsage: boolPointer(true), wantObfuscation: boolPointer(true),
		},
		{
			name:            "Responses explicit false",
			format:          llmprotocol.OpenAIResponsesV1,
			body:            `{"model":"m","input":"hello","stream":true,"stream_options":{"include_obfuscation":false}}`,
			wantObfuscation: boolPointer(false),
		},
		{
			name:            "Responses explicit true",
			format:          llmprotocol.OpenAIResponsesV1,
			body:            `{"model":"m","input":"hello","stream":true,"stream_options":{"include_obfuscation":true}}`,
			wantObfuscation: boolPointer(true),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request, _, _, err := engine.DecodeRequest(test.format, []byte(test.body))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(request.StreamOptions.IncludeUsage, test.wantUsage) ||
				!reflect.DeepEqual(request.StreamOptions.IncludeObfuscation, test.wantObfuscation) {
				t.Fatalf("stream options = %+v, want usage=%v obfuscation=%v", request.StreamOptions, test.wantUsage, test.wantObfuscation)
			}
			for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1} {
				if _, err := engine.TranslateRequest(test.format, target, []byte(test.body), func(request *llmprotocol.Request) error {
					request.Model = "routed-model"
					return nil
				}); err != nil {
					t.Fatalf("translate to %s: %v", target, err)
				}
			}
		})
	}
}

func TestOfficialStreamOptionsRequireStreaming(t *testing.T) {
	engine := NewBuiltinEngine()
	for name, test := range map[string]struct {
		format llmprotocol.WireFormat
		body   string
	}{
		"Chat": {
			format: llmprotocol.OpenAIChatV1,
			body:   `{"model":"m","messages":[{"role":"user","content":"hello"}],"stream_options":{"include_usage":true}}`,
		},
		"Responses": {
			format: llmprotocol.OpenAIResponsesV1,
			body:   `{"model":"m","input":"hello","stream_options":{"include_obfuscation":true}}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, _, err := engine.DecodeRequest(test.format, []byte(test.body))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != "stream_options_without_stream" {
				t.Fatalf("error = %T %v, want stream_options_without_stream", err, err)
			}
		})
	}
}

func TestOfficialStreamOptionsRejectUnknownFieldsAndInvalidTypesBeforeMutation(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{
			name: "Chat unknown option", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true,"future_option":true}}`,
		},
		{
			name: "Responses unknown option", format: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","input":"hello","stream":true,"stream_options":{"future_option":true}}`,
		},
		{
			name: "Chat invalid usage type", format: llmprotocol.OpenAIChatV1,
			body: `{"model":"m","messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":"yes"}}`,
		},
		{
			name: "Responses invalid obfuscation type", format: llmprotocol.OpenAIResponsesV1,
			body: `{"model":"m","input":"hello","stream":true,"stream_options":{"include_obfuscation":1}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, _, _, err := engine.DecodeRequestForMutation(test.format, []byte(test.body)); err == nil {
				t.Fatal("mutation-aware decode accepted an invalid stream option")
			}
		})
	}
}

func boolPointer(value bool) *bool {
	return &value
}

func TestOfficialIdentityStateAndReasoningFieldsTranslateSemantically(t *testing.T) {
	engine := NewBuiltinEngine()
	assertOfficialChatIdentityState(t, engine)
	assertOfficialResponsesIdentityState(t, engine)
	assertOfficialAnthropicIdentityState(t, engine)
}

func TestOfficialOmittedToolChoiceUsesAutomaticSelectionWhenToolsExist(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []struct {
		format llmprotocol.WireFormat
		body   string
	}{
		{
			format: llmprotocol.OpenAIChatV1,
			body:   `{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`,
		},
		{
			format: llmprotocol.OpenAIResponsesV1,
			body:   `{"model":"m","input":"hello","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`,
		},
		{
			format: llmprotocol.AnthropicMessagesV1,
			body:   `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hello"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}`,
		},
	}
	for _, test := range tests {
		t.Run(string(test.format), func(t *testing.T) {
			request, _, _, err := engine.DecodeRequest(test.format, []byte(test.body))
			if err != nil {
				t.Fatal(err)
			}
			if request.ToolChoice.Mode != llmprotocol.ToolChoiceAuto {
				t.Fatalf("tool choice = %q, want documented automatic default", request.ToolChoice.Mode)
			}
		})
	}
}

func TestOfficialStructuredOutputMatrixPreservesCompleteJSONSchema(t *testing.T) {
	engine := NewBuiltinEngine()
	strict := true
	wantSchema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)
	tests := []struct {
		name   string
		source llmprotocol.WireFormat
		target llmprotocol.WireFormat
		body   string
	}{
		{
			name:   "Chat to Responses",
			source: llmprotocol.OpenAIChatV1,
			target: llmprotocol.OpenAIResponsesV1,
			body:   `{"model":"m","messages":[{"role":"user","content":"answer"}],"response_format":{"type":"json_schema","json_schema":{"name":"answer","description":"one answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}}}`,
		},
		{
			name:   "Responses to Chat",
			source: llmprotocol.OpenAIResponsesV1,
			target: llmprotocol.OpenAIChatV1,
			body:   `{"model":"m","input":"answer","text":{"format":{"type":"json_schema","name":"answer","description":"one answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			translated, err := engine.TranslateRequest(test.source, test.target, []byte(test.body), func(request *llmprotocol.Request) error {
				request.Model = "routed-model"
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			decoded, _, _, err := engine.DecodeRequest(test.target, translated.Body)
			if err != nil {
				t.Fatal(err)
			}
			format := decoded.OutputFormat
			if format.Kind != llmprotocol.OutputJSONSchema || format.Name != "answer" ||
				format.Description != "one answer" || format.Strict == nil || *format.Strict != strict ||
				!jsonSemanticallyEqual(format.Schema, wantSchema) {
				t.Fatalf("structured output changed across %s: %+v body=%s", test.name, format, translated.Body)
			}
		})
	}
}

func TestOfficialAnthropicOutputConfigPreservesSchemaAndEffort(t *testing.T) {
	engine := NewBuiltinEngine()
	schema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`)
	body := []byte(`{
		"model":"m","max_tokens":64,
		"messages":[{"role":"user","content":"answer"}],
		"output_config":{
			"effort":"high",
			"format":{"type":"json_schema","schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}
		}
	}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	assertAnthropicOutputConfigRequest(t, request, schema)

	request.Generation++
	roundTrip, err := engine.EncodeRequest(llmprotocol.AnthropicMessagesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire anthropicRequestWire
	if err := json.Unmarshal(roundTrip.Body, &wire); err != nil {
		t.Fatal(err)
	}
	assertAnthropicOutputConfigWire(t, &wire, schema, roundTrip.Body)

	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(target), func(t *testing.T) {
			assertTranslatedAnthropicOutputConfig(t, engine, target, body, schema)
		})
	}
}

func assertAnthropicOutputConfigRequest(t *testing.T, request llmprotocol.Request, schema json.RawMessage) {
	t.Helper()
	format := request.OutputFormat
	if request.ReasoningEffort != "high" || format.Kind != llmprotocol.OutputJSONSchema ||
		format.Name != "structured_output" || format.Strict == nil || !*format.Strict {
		t.Fatalf("Anthropic output_config semantics = %+v", request)
	}
	if !jsonSemanticallyEqual(format.Schema, schema) {
		t.Fatalf("Anthropic output_config schema = %s", format.Schema)
	}
}

func assertAnthropicOutputConfigWire(
	t *testing.T,
	wire *anthropicRequestWire,
	schema json.RawMessage,
	body []byte,
) {
	t.Helper()
	if wire.OutputConfig == nil || wire.OutputConfig.Effort != "high" || wire.OutputConfig.Format == nil {
		t.Fatalf("Anthropic output_config round trip = %s", body)
	}
	if wire.OutputConfig.Format.Type != "json_schema" ||
		!jsonSemanticallyEqual(wire.OutputConfig.Format.Schema, schema) {
		t.Fatalf("Anthropic output_config format round trip = %s", body)
	}
}

func assertTranslatedAnthropicOutputConfig(
	t *testing.T,
	engine *Engine,
	target llmprotocol.WireFormat,
	body []byte,
	schema json.RawMessage,
) {
	t.Helper()
	translated, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1,
		target,
		body,
		func(request *llmprotocol.Request) error { request.Model = "routed-model"; return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, _, err := engine.DecodeRequest(target, translated.Body)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.ReasoningEffort != "high" || decoded.OutputFormat.Kind != llmprotocol.OutputJSONSchema ||
		!jsonSemanticallyEqual(decoded.OutputFormat.Schema, schema) {
		t.Fatalf("Anthropic output_config changed through %s: %+v body=%s", target, decoded, translated.Body)
	}
}

func TestAnthropicRejectsOpenAIOnlyReasoningEfforts(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, effort := range []string{"none", "minimal"} {
		t.Run(effort, func(t *testing.T) {
			body := []byte(`{"model":"m","messages":[{"role":"user","content":"answer"}],"reasoning_effort":"` + effort + `"}`)
			_, err := engine.TranslateRequest(
				llmprotocol.OpenAIChatV1,
				llmprotocol.AnthropicMessagesV1,
				body,
				nil,
			)
			assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "lossy_translation")
		})
	}
}

func TestOfficialAnthropicTerminalReasonInventoryIsClosed(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []anthropicTerminalReasonCase{
		{reason: "end_turn", neutral: llmprotocol.StopEndTurn},
		{reason: "max_tokens", neutral: llmprotocol.StopMaxTokens},
		{reason: "stop_sequence", neutral: llmprotocol.StopSequence, openAIClosed: true},
		{reason: "tool_use", neutral: llmprotocol.StopToolCall},
		{reason: "pause_turn", neutral: llmprotocol.StopPaused, openAIClosed: true},
		{reason: "refusal", neutral: llmprotocol.StopContentFilter},
		{reason: "model_context_window_exceeded", neutral: llmprotocol.StopContextWindow, openAIClosed: true},
	}
	for _, test := range tests {
		t.Run(test.reason, func(t *testing.T) { assertAnthropicTerminalReason(t, engine, test) })
	}
}

type anthropicTerminalReasonCase struct {
	reason       string
	neutral      llmprotocol.StopReason
	openAIClosed bool
}

func assertAnthropicTerminalReason(t *testing.T, engine *Engine, test anthropicTerminalReasonCase) {
	t.Helper()
	stopSequence := "null"
	if test.reason == "stop_sequence" {
		stopSequence = `"END"`
	}
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"done"}],"stop_reason":"` + test.reason + `","stop_sequence":` + stopSequence + `,"usage":{"input_tokens":1,"output_tokens":1}}`)
	decoded, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.StopReason != test.neutral {
		t.Fatalf("decoded stop reason = %q, want %q", decoded.StopReason, test.neutral)
	}
	if test.reason == "stop_sequence" && decoded.MatchedStopSequence != "END" {
		t.Fatalf("matched stop sequence = %q, want END", decoded.MatchedStopSequence)
	}
	assertAnthropicTerminalRoundTrip(t, engine, test.reason, body)
	if test.openAIClosed {
		assertAnthropicTerminalClosedForOpenAI(t, engine, test.reason, body)
	}
}

func assertAnthropicTerminalRoundTrip(t *testing.T, engine *Engine, reason string, body []byte) {
	t.Helper()
	roundTrip, err := engine.TranslateResponse(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.AnthropicMessagesV1,
		body,
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	var wire anthropicResponseWire
	if err := json.Unmarshal(roundTrip.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.StopReason == nil || *wire.StopReason != reason {
		t.Fatalf("round-trip stop reason = %v, body=%s", wire.StopReason, roundTrip.Body)
	}
	if reason == "stop_sequence" && (wire.StopSequence == nil || *wire.StopSequence != "END") {
		t.Fatalf("round-trip matched stop sequence = %v, body=%s", wire.StopSequence, roundTrip.Body)
	}
}

func assertAnthropicTerminalClosedForOpenAI(t *testing.T, engine *Engine, reason string, body []byte) {
	t.Helper()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		if _, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, body, nil); err == nil {
			t.Fatalf("%s silently accepted unrepresentable stop reason %q", target, reason)
		}
	}
}

func TestOfficialResponsesContentFilterIncompleteRoundTrip(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"resp_1","object":"response","model":"m","status":"incomplete","output":[],"incomplete_details":{"reason":"content_filter"},"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	translated, err := engine.TranslateResponse(
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.OpenAIResponsesV1,
		body,
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if translated.Response.StopReason != llmprotocol.StopContentFilter {
		t.Fatalf("decoded stop reason = %q", translated.Response.StopReason)
	}
	var wire responsesResponseWire
	if err := json.Unmarshal(translated.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Status != "incomplete" || wire.IncompleteDetails == nil || wire.IncompleteDetails.Reason != "content_filter" {
		t.Fatalf("content-filter terminal state changed: %s", translated.Body)
	}
}

func TestOfficialEncodedResponseResourcesContainRequiredFields(t *testing.T) {
	engine := NewBuiltinEngine()
	chat := []byte(`{"id":"response_1","object":"chat.completion","created":100,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	assertEncodedResponsesResource(t, engine, chat)
	assertEncodedChatResource(t, engine)
	assertEncodedAnthropicResource(t, engine, chat)
}

func assertEncodedResponsesResource(t *testing.T, engine *Engine, chat []byte) {
	t.Helper()
	responses, err := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		chat,
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	var response map[string]json.RawMessage
	if err := json.Unmarshal(responses.Body, &response); err != nil {
		t.Fatal(err)
	}
	assertJSONFields(t, response, []string{
		"created_at", "error", "id", "incomplete_details", "instructions", "metadata", "model",
		"object", "output", "parallel_tool_calls", "temperature", "tool_choice", "tools", "top_p",
	}, "official Responses resource", responses.Body)
	var output []map[string]json.RawMessage
	if err := json.Unmarshal(response["output"], &output); err != nil || len(output) != 1 {
		t.Fatalf("official Responses output is invalid: %v body=%s", err, responses.Body)
	}
	assertJSONFields(
		t, output[0], []string{"content", "id", "role", "status", "type"},
		"official Responses output message", responses.Body,
	)
	assertResponsesUsageFields(t, response["usage"])
}

func assertResponsesUsageFields(t *testing.T, body json.RawMessage) {
	t.Helper()
	var usage struct {
		InputDetails  map[string]json.RawMessage `json:"input_tokens_details"`
		OutputDetails map[string]json.RawMessage `json:"output_tokens_details"`
	}
	if err := json.Unmarshal(body, &usage); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"cached_tokens", "cache_write_tokens"} {
		if _, found := usage.InputDetails[field]; !found {
			t.Errorf("Responses input usage omitted required field %q", field)
		}
	}
	if _, found := usage.OutputDetails["reasoning_tokens"]; !found {
		t.Error("Responses output usage omitted required reasoning_tokens")
	}
}

func assertJSONFields(
	t *testing.T,
	object map[string]json.RawMessage,
	fields []string,
	label string,
	body []byte,
) {
	t.Helper()
	for _, field := range fields {
		if _, found := object[field]; !found {
			t.Errorf("%s omitted required field %q: %s", label, field, body)
		}
	}
}

func assertEncodedChatResource(t *testing.T, engine *Engine) {
	t.Helper()
	translatedChat, err := engine.TranslateResponse(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIChatV1,
		[]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`),
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	var chatResource map[string]json.RawMessage
	if err := json.Unmarshal(translatedChat.Body, &chatResource); err != nil {
		t.Fatal(err)
	}
	if _, found := chatResource["created"]; !found {
		t.Fatalf("official Chat resource omitted required created field: %s", translatedChat.Body)
	}
}

func assertEncodedAnthropicResource(t *testing.T, engine *Engine, chat []byte) {
	t.Helper()
	translatedAnthropic, err := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.AnthropicMessagesV1,
		chat,
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	var anthropicResource map[string]json.RawMessage
	if err := json.Unmarshal(translatedAnthropic.Body, &anthropicResource); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{
		"container", "content", "id", "model", "role", "stop_details", "stop_reason",
		"stop_sequence", "type", "usage",
	} {
		if _, found := anthropicResource[field]; !found {
			t.Errorf("official Anthropic resource omitted required field %q: %s", field, translatedAnthropic.Body)
		}
	}
	var anthropicUsage map[string]json.RawMessage
	if err := json.Unmarshal(anthropicResource["usage"], &anthropicUsage); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{
		"cache_creation", "cache_creation_input_tokens", "cache_read_input_tokens", "inference_geo",
		"input_tokens", "output_tokens", "output_tokens_details", "server_tool_use", "service_tier",
	} {
		if _, found := anthropicUsage[field]; !found {
			t.Errorf("official Anthropic usage omitted required field %q: %s", field, translatedAnthropic.Body)
		}
	}
}

func jsonSemanticallyEqual(left, right []byte) bool {
	var leftValue, rightValue any
	return json.Unmarshal(left, &leftValue) == nil && json.Unmarshal(right, &rightValue) == nil &&
		reflect.DeepEqual(leftValue, rightValue)
}

func assertOfficialChatIdentityState(t *testing.T, engine *Engine) {
	t.Helper()
	chatBody := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}],"store":true,"user":"user-1"}`)
	chat, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, chatBody)
	if err != nil {
		t.Fatal(err)
	}
	if chat.Store == nil || !*chat.Store || chat.EndUserID != "user-1" {
		t.Fatalf("Chat identity or store semantics were lost: %+v", chat)
	}
	translated, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, chatBody, nil)
	if err != nil {
		t.Fatal(err)
	}
	var responses map[string]any
	if json.Unmarshal(translated.Body, &responses) != nil || responses["store"] != true || responses["user"] != "user-1" {
		t.Fatalf("Chat to Responses identity/state translation = %s", translated.Body)
	}
}

func assertOfficialResponsesIdentityState(t *testing.T, engine *Engine) {
	t.Helper()
	responsesBody := []byte(`{"model":"m","input":"hello","conversation":{"id":"conv-1"},"reasoning":{"effort":"high"},"truncation":"auto","user":"user-2"}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, responsesBody)
	if err != nil {
		t.Fatal(err)
	}
	if request.ConversationID != "conv-1" || request.ReasoningEffort != "high" || request.Truncation != "auto" || request.EndUserID != "user-2" {
		t.Fatalf("Responses semantic fields were lost: %+v", request)
	}
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip map[string]any
	if json.Unmarshal(encoded.Body, &roundTrip) != nil || roundTrip["conversation"] != "conv-1" || roundTrip["truncation"] != "auto" || roundTrip["user"] != "user-2" {
		t.Fatalf("Responses semantic re-encoding = %s", encoded.Body)
	}
}

func assertOfficialAnthropicIdentityState(t *testing.T, engine *Engine) {
	t.Helper()
	anthropicBody := []byte(`{"model":"m","max_tokens":16,"metadata":{"user_id":"user-3"},"messages":[{"role":"user","content":"hello"}]}`)
	toChat, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, anthropicBody, nil)
	if err != nil {
		t.Fatal(err)
	}
	var chatWire map[string]any
	if json.Unmarshal(toChat.Body, &chatWire) != nil || chatWire["user"] != "user-3" {
		t.Fatalf("Messages to Chat end-user translation = %s", toChat.Body)
	}
}

func TestOfficialProviderResponseShapesDecodeWithoutSilentLoss(t *testing.T) {
	engine := NewBuiltinEngine()
	assertOfficialResponsesProviderShape(t, engine)
	assertOfficialAnthropicProviderShape(t, engine)
}

func assertOfficialResponsesProviderShape(t *testing.T, engine *Engine) {
	t.Helper()
	responsesBody := []byte(`{
		"id":"resp_1","object":"response","created_at":1,"completed_at":2,"status":"completed",
		"error":null,"incomplete_details":null,"instructions":"answer","max_output_tokens":64,
		"model":"m","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],
		"parallel_tool_calls":true,"previous_response_id":"resp_0","reasoning":{"effort":"high"},
		"store":true,"temperature":0.2,"text":{"format":{"type":"text"}},"tool_choice":"auto",
		"tools":[],"top_p":0.9,"truncation":"disabled","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8},
		"user":"user-1","metadata":{"trace":"1"}
	}`)
	response, _, diagnostics, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, responsesBody)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Output) != 1 || response.Output[0].Content[0].Text != "done" || len(diagnostics) == 0 {
		t.Fatalf("Responses output or explicit omission diagnostics missing: response=%+v diagnostics=%+v", response, diagnostics)
	}
}

func assertOfficialAnthropicProviderShape(t *testing.T, engine *Engine) {
	t.Helper()
	anthropicBody := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-test","container":{},
		"content":[{"type":"text","text":"done"}],"stop_details":null,"stop_reason":"end_turn","stop_sequence":null,
		"usage":{"cache_creation":{"ephemeral_1h_input_tokens":1,"ephemeral_5m_input_tokens":2},
		"cache_creation_input_tokens":3,"cache_read_input_tokens":4,"inference_geo":"us","input_tokens":10,
		"output_tokens":6,"output_tokens_details":{"thinking_tokens":2},"server_tool_use":{"web_search_requests":1},
		"service_tier":"standard"}
	}`)
	response, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, anthropicBody)
	if err != nil {
		t.Fatal(err)
	}
	if tokenValue(response.Usage.InputUncached) != 10 || tokenValue(response.Usage.InputCacheWrite) != 3 ||
		tokenValue(response.Usage.InputCacheRead) != 4 || tokenValue(response.Usage.InputTotal) != 17 ||
		tokenValue(response.Usage.OutputReasoning) != 2 || tokenValue(response.Usage.OutputOther) != 4 ||
		tokenValue(response.Usage.Total) != 23 || len(diagnostics) == 0 {
		t.Fatalf("Anthropic accounting or explicit omission diagnostics missing: usage=%+v diagnostics=%+v", response.Usage, diagnostics)
	}
}

func TestOfficialResponseMetadataIsExplicitlyDiagnosed(t *testing.T) {
	engine := NewBuiltinEngine()
	chatBody := []byte(`{
		"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],
		"metadata":{"trace":"1"},"moderation":{"status":"passed"},
		"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3,"compute_units":4}
	}`)
	_, _, diagnostics, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, chatBody)
	if err != nil {
		t.Fatal(err)
	}
	assertDiagnosticFields(t, diagnostics, "metadata", "moderation", "usage.compute_units")

	responsesBody := []byte(`{
		"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m",
		"output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"done","annotations":[],"logprobs":[]}]}],
		"background":false,"conversation":{"id":"conv_1"},"max_tool_calls":4,"moderation":{"status":"passed"},
		"output_text":"done","prompt":{"id":"pmpt_1"},"prompt_cache_key":"cache_1",
		"prompt_cache_options":{"retention":"24h"},"prompt_cache_retention":"24h",
		"safety_identifier":"safe_1","service_tier":"default","top_logprobs":0,
		"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3,"compute_units":4}
	}`)
	_, _, diagnostics, err = engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, responsesBody)
	if err != nil {
		t.Fatal(err)
	}
	assertDiagnosticFields(
		t, diagnostics, "background", "conversation", "max_tool_calls", "moderation", "output.content.logprobs", "output.phase", "output_text",
		"output.status", "prompt", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention",
		"safety_identifier", "service_tier", "top_logprobs", "usage.compute_units",
	)
}

func assertDiagnosticFields(t *testing.T, diagnostics llmprotocol.Diagnostics, expected ...string) {
	t.Helper()
	actual := make([]string, 0, len(diagnostics))
	for _, diagnostic := range diagnostics {
		actual = append(actual, diagnostic.Field)
	}
	sort.Strings(actual)
	sort.Strings(expected)
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("diagnostic fields = %v, want %v", actual, expected)
	}
}

func assertProtocolError(t *testing.T, err error, category llmprotocol.ErrorCategory, code string) {
	t.Helper()
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != category || code != "" && protocolError.Code != code {
		t.Fatalf("returned %T %v, want %s/%s", err, err, category, code)
	}
}

func fields(values ...string) []string {
	result := append([]string(nil), values...)
	sort.Strings(result)
	return result
}

func assertClosedDiscriminatorInventory(
	t *testing.T,
	name string,
	officialCount int,
	supported []string,
	unsupported []string,
) {
	t.Helper()
	seen := make(map[string]string, officialCount)
	for _, group := range []struct {
		name   string
		values []string
	}{
		{name: "supported", values: supported},
		{name: "unsupported", values: unsupported},
	} {
		for _, value := range group.values {
			if previous, ok := seen[value]; ok {
				t.Fatalf("%s discriminator %q appears in both %s and %s inventories", name, value, previous, group.name)
			}
			seen[value] = group.name
		}
	}
	if len(seen) != officialCount {
		t.Fatalf("%s discriminator inventory has %d unique variants, want %d", name, len(seen), officialCount)
	}
}

func assertClosedFieldDisposition(
	t *testing.T,
	name string,
	official []string,
	dispositions map[string][]string,
) {
	t.Helper()
	seen := make(map[string]string, len(official))
	for disposition, values := range dispositions {
		for _, value := range values {
			if previous, ok := seen[value]; ok {
				t.Fatalf("%s field %q appears in both %s and %s dispositions", name, value, previous, disposition)
			}
			seen[value] = disposition
		}
	}
	actual := make([]string, 0, len(seen))
	for field := range seen {
		actual = append(actual, field)
	}
	sort.Strings(actual)
	if !reflect.DeepEqual(actual, official) {
		t.Fatalf("%s field dispositions are incomplete\n got: %v\nwant: %v", name, actual, official)
	}
}

func jsonFieldNames(value reflect.Type) []string {
	result := make([]string, 0, value.NumField())
	for index := 0; index < value.NumField(); index++ {
		name := value.Field(index).Tag.Get("json")
		if comma := len(name); comma > 0 {
			for offset, char := range name {
				if char == ',' {
					name = name[:offset]
					break
				}
			}
		}
		if name != "" && name != "-" {
			result = append(result, name)
		}
	}
	sort.Strings(result)
	return result
}

func cloneJSONMap(source map[string]any) map[string]any {
	result := make(map[string]any, len(source)+1)
	for key, value := range source {
		result[key] = value
	}
	return result
}
