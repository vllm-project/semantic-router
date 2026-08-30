package protocolcodec

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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
		"computer_use_preview", "custom", "file_search", "mcp",
		"programmatic_tool_calling", "shell", "web_search_preview", "web_search_preview_2025_03_11",
	)
	assertClosedDiscriminatorInventory(
		t,
		"OpenAI Responses object tool choice",
		15,
		fields("function", "image_generation"),
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
	t.Run("responses/image_generation", func(t *testing.T) {
		body := []byte(`{"model":"m","input":"hello","tools":[{"type":"image_generation"}],"tool_choice":{"type":"image_generation"}}`)
		request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
		if err != nil || request.ImageGeneration == nil || request.ToolChoice.Mode != llmprotocol.ToolChoiceImageGeneration {
			t.Fatalf("image-generation tool choice = %+v, %v", request.ToolChoice, err)
		}
	})
	t.Run("responses/image_generation rejects function payload", func(t *testing.T) {
		body := []byte(`{"model":"m","input":"hello","tools":[{"type":"image_generation"}],"tool_choice":{"type":"image_generation","name":"lookup"}}`)
		_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
		assertProtocolError(t, err, llmprotocol.ErrorInvalidRequest, "invalid_json")
	})
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
