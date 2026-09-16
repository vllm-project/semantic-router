package protocolcodec

import (
	"encoding/json"
	"errors"
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
	supported := fields("function_call", "function_call_output", "image_generation_call", "item_reference", "message", "reasoning")
	unsupported := fields(
		"additional_tools", "apply_patch_call", "apply_patch_call_output", "code_interpreter_call",
		"compaction", "compaction_trigger", "computer_call", "computer_call_output", "custom_tool_call",
		"custom_tool_call_output", "file_search_call", "function_shell_call",
		"function_shell_call_output", "local_shell_call", "local_shell_call_output",
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
		"function_shell_call_output", "local_shell_call", "local_shell_call_output",
		"mcp_approval_request", "mcp_approval_response", "mcp_call", "mcp_list_tools", "program",
		"program_output", "tool_search_call", "tool_search_output", "web_search_call",
	)
	assertClosedDiscriminatorInventory(
		t,
		"OpenAI Responses output item",
		28,
		fields("function_call", "image_generation_call", "message", "reasoning"),
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
		"file_search", "local_shell", "mcp", "namespace",
		"programmatic_tool_calling", "shell", "tool_search", "web_search", "web_search_preview",
	)
	assertClosedDiscriminatorInventory(t, "OpenAI Responses tool", 16, fields("function", "image_generation"), unsupported)
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
