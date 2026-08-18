package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/anthropic"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestApplyDecisionToolHistoryPolicyStripsModernAndLegacyToolHistory(t *testing.T) {
	request := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("Keep this system instruction."),
			openai.UserMessage("Inspect the service."),
			openai.AssistantMessage("Keep this ordinary answer."),
			toolHistoryAssistantCallMessage("lookup_service"),
			openai.ToolMessage("sensitive lookup output", "call-lookup_service"),
			legacyAssistantFunctionCallMessage("legacy_lookup"),
			openai.ChatCompletionMessageParamOfFunction("sensitive legacy output", "legacy_lookup"),
			openai.UserMessage("Give me a safe summary."),
		},
	}
	removed := applyToolsPluginRequestPolicy(request, &config.ToolsPluginConfig{
		Enabled: true, Mode: config.ToolsPluginModeNone, StripToolHistory: true,
	})

	if removed != 4 {
		t.Fatalf("removed = %d, want 4", removed)
	}
	if len(request.Messages) != 4 {
		t.Fatalf("messages = %d, want 4", len(request.Messages))
	}
	if request.Messages[0].OfSystem == nil || request.Messages[1].OfUser == nil ||
		request.Messages[2].OfAssistant == nil || request.Messages[3].OfUser == nil {
		t.Fatalf("retained message roles are incorrect: %#v", request.Messages)
	}
	if got := extractAssistantMessageContent(request.Messages[2].OfAssistant.Content); got != "Keep this ordinary answer." {
		t.Fatalf("ordinary assistant content = %q", got)
	}
}

func TestApplyDecisionToolHistoryPolicyPreservesHistoryWhenDisabled(t *testing.T) {
	request := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			toolHistoryAssistantCallMessage("lookup_service"),
			openai.ToolMessage("lookup output", "call-lookup_service"),
		},
	}
	if removed := applyToolsPluginRequestPolicy(request, &config.ToolsPluginConfig{
		Enabled: true, Mode: config.ToolsPluginModeNone,
	}); removed != 0 {
		t.Fatalf("removed = %d, want 0", removed)
	}
	if len(request.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(request.Messages))
	}
}

func TestApplyDecisionToolHistoryPolicyPreservesMixedAssistantContent(t *testing.T) {
	mixed := openai.AssistantMessage("Keep this explanation.")
	mixed.OfAssistant.ToolCalls = toolHistoryAssistantCallMessage("lookup_service").OfAssistant.ToolCalls
	request := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{mixed},
	}

	removed := applyToolsPluginRequestPolicy(request, &config.ToolsPluginConfig{
		Enabled: true, Mode: config.ToolsPluginModeNone, StripToolHistory: true,
	})
	if removed != 1 {
		t.Fatalf("removed = %d, want 1", removed)
	}
	if len(request.Messages) != 1 || request.Messages[0].OfAssistant == nil {
		t.Fatalf("mixed assistant message was removed: %#v", request.Messages)
	}
	assistant := request.Messages[0].OfAssistant
	if got := extractAssistantMessageContent(assistant.Content); got != "Keep this explanation." {
		t.Fatalf("assistant content = %q", got)
	}
	if len(assistant.ToolCalls) != 0 {
		t.Fatalf("assistant tool calls were retained: %#v", assistant.ToolCalls)
	}
}

func TestApplyPreDispatchToolsPolicyClearsControlsAndPreservesOpaqueFields(t *testing.T) {
	raw := []byte(`{
		"model":"vllm-sr/vault",
		"messages":[
			{"role":"user","content":"Inspect the service."},
			{"role":"assistant","content":"Keep this explanation.","reasoning_content":"opaque reasoning","cache_control":{"type":"ephemeral"},"opaque_message_integer":9007199254740995,"tool_calls":[{"id":"call-lookup","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call-lookup","content":"sensitive output"},
			{"role":"user","content":"Give me a safe summary."}
		],
		"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
		"tool_choice":"auto",
		"functions":[{"name":"legacy_lookup","parameters":{"type":"object"}}],
		"function_call":"auto",
		"parallel_tool_calls":true,
		"web_search_options":{"search_context_size":"low"},
		"opaque_integer":9007199254740993
	}`)
	request, err := parseOpenAIRequest(raw)
	if err != nil {
		t.Fatalf("parseOpenAIRequest: %v", err)
	}
	ctx := toolsHistoryTestContext(t, true)
	ctx.OriginalRequestBody = raw
	router := &OpenAIRouter{}

	changed, err := router.applyPreDispatchToolsPolicy(request, ctx)
	if err != nil {
		t.Fatalf("applyPreDispatchToolsPolicy: %v", err)
	}
	if !changed {
		t.Fatal("policy was not applied")
	}
	assertFilteredOpenAIRequestMessages(t, request.Messages)
	body := decodeRawJSONObject(t, ctx.workingRequestBody(), "working body")
	assertRawFieldsAbsent(t, body, []string{
		"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls", "web_search_options",
	})
	assertRawJSONEquals(t, body, "opaque_integer", "9007199254740993")
	assertOpaqueAssistantFields(t, decodeRawJSONMessages(t, body["messages"], "messages"))
}

func TestApplyPreDispatchToolsPolicyUpdatesResponsesAPITranslatedBody(t *testing.T) {
	original := []byte(`{
		"model":"vllm-sr/vault",
		"input":[
			{"type":"function_call","call_id":"call-lookup","name":"lookup","arguments":"{}"},
			{"type":"function_call_output","call_id":"call-lookup","output":"sensitive output"},
			{"type":"message","role":"user","content":"Give me a safe summary."}
		]
	}`)
	translated := []byte(`{
		"model":"vllm-sr/vault",
		"messages":[
			{"role":"assistant","content":"Keep this explanation.","reasoning_content":"opaque reasoning","cache_control":{"type":"ephemeral"},"tool_calls":[{"id":"call-lookup","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call-lookup","content":"sensitive output"},
			{"role":"user","content":"Give me a safe summary."}
		],
		"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]
	}`)
	request, err := parseOpenAIRequest(translated)
	if err != nil {
		t.Fatalf("parseOpenAIRequest: %v", err)
	}
	ctx := toolsHistoryTestContext(t, true)
	ctx.OriginalRequestBody = original
	ctx.WorkingRequestBody = translated
	ctx.ResponseAPICtx = &ResponseAPIContext{
		IsResponseAPIRequest: true,
		TranslatedBody:       translated,
	}

	changed, err := (&OpenAIRouter{}).applyPreDispatchToolsPolicy(request, ctx)
	if err != nil {
		t.Fatalf("applyPreDispatchToolsPolicy: %v", err)
	}
	if !changed {
		t.Fatal("policy was not applied")
	}
	body := decodeRawJSONObject(t, ctx.ResponseAPICtx.TranslatedBody, "translated body")
	assertRawFieldsAbsent(t, body, []string{"tools"})
	assertResponsesAPIFilteredMessages(t, decodeRawJSONMessages(t, body["messages"], "translated messages"))
}

func TestHandleToolSelectionPatchesFinalBodyWithoutDroppingOtherPlugins(t *testing.T) {
	request := &openai.ChatCompletionNewParams{
		Model: "provider/model",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Give me a safe summary."),
		},
	}
	ctx := toolsHistoryTestContext(t, true)
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{
				BodyMutation: &ext_proc.BodyMutation{
					Mutation: &ext_proc.BodyMutation_Body{Body: []byte(`{
						"model":"provider/model",
						"messages":[
							{"role":"system","content":"Keep the injected system prompt."},
							{"role":"user","content":"Give me a safe summary."}
						],
						"reasoning_effort":"high",
						"max_completion_tokens":512,
						"tools":[{"type":"function","function":{"name":"lookup"}}]
					}`)},
				},
			}},
		},
	}

	(&OpenAIRouter{Config: &config.RouterConfig{}}).handleToolSelectionForRequest(request, response, ctx)

	mutated := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	var body map[string]json.RawMessage
	if err := json.Unmarshal(mutated, &body); err != nil {
		t.Fatalf("decode mutated body: %v", err)
	}
	if _, exists := body["tools"]; exists {
		t.Error("tools were not removed")
	}
	if string(body["reasoning_effort"]) != `"high"` || string(body["max_completion_tokens"]) != "512" {
		t.Fatalf("non-tool mutations were lost: %s", mutated)
	}
	var messages []map[string]interface{}
	if err := json.Unmarshal(body["messages"], &messages); err != nil {
		t.Fatalf("decode messages: %v", err)
	}
	if len(messages) != 2 || messages[0]["role"] != "system" {
		t.Fatalf("injected system prompt was lost: %#v", messages)
	}
}

func TestModifyRequestBodyForAutoRoutingPreservesNestedProviderFields(t *testing.T) {
	raw := []byte(`{
		"model":"vllm-sr/vault",
		"messages":[
			{"role":"assistant","content":"Keep this explanation.","reasoning_content":"opaque reasoning","cache_control":{"type":"ephemeral"},"opaque_message_integer":9007199254740995},
			{"role":"user","content":"Give me a safe summary."}
		]
	}`)
	request, err := parseOpenAIRequest(raw)
	if err != nil {
		t.Fatalf("parseOpenAIRequest: %v", err)
	}
	decision := decisionWithSystemPrompt("vault_private_default", "Treat inputs as confidential.")
	ctx := &RequestContext{
		OriginalRequestBody:     raw,
		ExpectStreamingResponse: true,
		VSRSelectedDecision:     &decision,
	}
	mutated, err := (&OpenAIRouter{}).modifyRequestBodyForAutoRouting(
		request,
		"provider/final-model",
		decision.Name,
		false,
		nil,
		ctx,
	)
	if err != nil {
		t.Fatalf("modifyRequestBodyForAutoRouting: %v", err)
	}
	assertAutoRoutingProviderFields(t, decodeRawJSONObject(t, mutated, "mutated body"))
}

func TestApplyPreDispatchToolsPolicyStripsAnthropicToolUseAndResults(t *testing.T) {
	raw := []byte(`{
		"model":"vllm-sr/vault",
		"max_tokens":256,
		"messages":[
			{"role":"user","content":"Inspect the service."},
			{"role":"assistant","content":[{"type":"text","text":"Keep this explanation."},{"type":"tool_use","id":"tu-1","name":"lookup","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu-1","content":"sensitive output"}]},
			{"role":"user","content":[
				{"type":"text","text":"Give me a safe summary."},
				{"type":"image","source":{"type":"url","url":"https://example.com/current.png"},"cache_control":{"type":"ephemeral"}}
			]}
		],
		"tools":[{"name":"lookup","description":"lookup","input_schema":{"type":"object"}}]
	}`)
	ctx := toolsHistoryTestContext(t, true)
	ctx.ClientProtocol = config.ClientProtocolAnthropic
	ctx.OriginalRequestBody = raw
	request, err := parseRequestForProtocol(ctx, raw)
	if err != nil {
		t.Fatalf("parseRequestForProtocol: %v", err)
	}

	changed, err := (&OpenAIRouter{}).applyPreDispatchToolsPolicy(request, ctx)
	if err != nil {
		t.Fatalf("applyPreDispatchToolsPolicy: %v", err)
	}
	if !changed {
		t.Fatal("policy was not applied")
	}
	if !ctx.requestBodyMutated() {
		t.Fatal("native Anthropic body was not synchronized with the filtered canonical IR")
	}
	assertTextDoesNotContainAny(t, ctx.workingRequestBody(), []string{`"tool_use"`, `"tool_result"`, `"tools"`}, "filtered native Anthropic body")
	if !strings.Contains(string(ctx.workingRequestBody()), `"text":"Keep this explanation."`) {
		t.Fatalf("mixed Anthropic assistant text was lost: %s", ctx.workingRequestBody())
	}
	passthrough, err := anthropic.BuildPassthroughFromAnthropicBody(ctx.workingRequestBody())
	if err != nil {
		t.Fatalf("BuildPassthroughFromAnthropicBody: %v", err)
	}
	// prepareAnthropicRoutingRequest clears this map for native Anthropic
	// requests because the canonical IR already carries the image.
	passthrough.UserMessageImageBlocks = nil
	outbound, err := anthropic.ToAnthropicRequestBodyWithPassthrough(request, passthrough)
	if err != nil {
		t.Fatalf("ToAnthropicRequestBodyWithPassthrough: %v", err)
	}
	assertTextDoesNotContainAny(t, outbound, []string{`"tool_use"`, `"tool_result"`, `"tools"`}, "outbound Anthropic body")
	if got := strings.Count(string(outbound), `"type":"image"`); got != 1 {
		t.Fatalf("outbound image count = %d, want 1: %s", got, outbound)
	}
	if !strings.Contains(string(outbound), `"url":"https://example.com/current.png"`) ||
		!strings.Contains(string(outbound), `"cache_control":{"type":"ephemeral"}`) {
		t.Fatalf("current image or its cache control was lost/misaligned: %s", outbound)
	}
	if !strings.Contains(string(outbound), `"text":"Keep this explanation."`) {
		t.Fatalf("outbound mixed Anthropic assistant text was lost: %s", outbound)
	}
}

func decodeRawJSONObject(t *testing.T, body []byte, label string) map[string]json.RawMessage {
	t.Helper()
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode %s: %v", label, err)
	}
	return decoded
}

func decodeRawJSONMessages(t *testing.T, body []byte, label string) []map[string]json.RawMessage {
	t.Helper()
	var decoded []map[string]json.RawMessage
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode %s: %v", label, err)
	}
	return decoded
}

func assertRawFieldsAbsent(t *testing.T, object map[string]json.RawMessage, fields []string) {
	t.Helper()
	for _, field := range fields {
		if _, exists := object[field]; exists {
			t.Errorf("%s was not removed", field)
		}
	}
}

func assertRawJSONEquals(t *testing.T, object map[string]json.RawMessage, field, want string) {
	t.Helper()
	if got := string(object[field]); got != want {
		t.Fatalf("%s = %s, want %s", field, got, want)
	}
}

func assertFilteredOpenAIRequestMessages(t *testing.T, messages []openai.ChatCompletionMessageParamUnion) {
	t.Helper()
	if len(messages) != 3 || messages[0].OfUser == nil || messages[1].OfAssistant == nil || messages[2].OfUser == nil {
		t.Fatalf("provider-bound messages retain tool history: %#v", messages)
	}
	if got := extractAssistantMessageContent(messages[1].OfAssistant.Content); got != "Keep this explanation." {
		t.Fatalf("mixed assistant content = %q", got)
	}
}

func assertOpaqueAssistantFields(t *testing.T, messages []map[string]json.RawMessage) {
	t.Helper()
	if len(messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(messages))
	}
	assertRawFieldsAbsent(t, messages[1], []string{"tool_calls", "function_call"})
	assertRawJSONEquals(t, messages[1], "reasoning_content", `"opaque reasoning"`)
	assertRawJSONEquals(t, messages[1], "cache_control", `{"type":"ephemeral"}`)
	assertRawJSONEquals(t, messages[1], "opaque_message_integer", "9007199254740995")
}

func assertResponsesAPIFilteredMessages(t *testing.T, messages []map[string]json.RawMessage) {
	t.Helper()
	if len(messages) != 2 || string(messages[0]["role"]) != `"assistant"` || string(messages[1]["role"]) != `"user"` {
		t.Fatalf("Responses API translated body retained tool history: %#v", messages)
	}
	assertRawFieldsAbsent(t, messages[0], []string{"tool_calls"})
	assertRawJSONEquals(t, messages[0], "content", `"Keep this explanation."`)
	assertRawJSONEquals(t, messages[0], "reasoning_content", `"opaque reasoning"`)
	assertRawJSONEquals(t, messages[0], "cache_control", `{"type":"ephemeral"}`)
}

func assertAutoRoutingProviderFields(t *testing.T, body map[string]json.RawMessage) {
	t.Helper()
	assertRawJSONEquals(t, body, "model", `"provider/final-model"`)
	assertRawJSONEquals(t, body, "stream", "true")
	streamOptions := decodeRawJSONObject(t, body["stream_options"], "stream_options")
	assertRawJSONEquals(t, streamOptions, "include_usage", "true")
	messages := decodeRawJSONMessages(t, body["messages"], "messages")
	if len(messages) != 3 || string(messages[0]["role"]) != `"system"` ||
		string(messages[0]["content"]) != `"Treat inputs as confidential."` ||
		string(messages[1]["reasoning_content"]) != `"opaque reasoning"` ||
		string(messages[1]["cache_control"]) != `{"type":"ephemeral"}` ||
		string(messages[1]["opaque_message_integer"]) != "9007199254740995" {
		t.Fatalf("nested provider fields were lost: %#v", messages)
	}
}

func assertTextDoesNotContainAny(t *testing.T, body []byte, forbidden []string, label string) {
	t.Helper()
	for _, value := range forbidden {
		if strings.Contains(string(body), value) {
			t.Fatalf("%s retained %s: %s", label, value, body)
		}
	}
}

func toolsHistoryTestContext(t *testing.T, strip bool) *RequestContext {
	t.Helper()
	payload, err := config.NewStructuredPayload(config.ToolsPluginConfig{
		Enabled:          true,
		Mode:             config.ToolsPluginModeNone,
		StripToolHistory: strip,
	})
	if err != nil {
		t.Fatalf("NewStructuredPayload: %v", err)
	}
	return &RequestContext{VSRSelectedDecision: &config.Decision{
		Name: "vault_private_default",
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginTools, Configuration: payload},
		},
	}}
}

func legacyAssistantFunctionCallMessage(name string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfAssistant: &openai.ChatCompletionAssistantMessageParam{
			FunctionCall: openai.ChatCompletionAssistantMessageParamFunctionCall{
				Name:      name,
				Arguments: `{}`,
			},
		},
	}
}

func toolHistoryAssistantCallMessage(name string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionMessageParamUnion{
		OfAssistant: &openai.ChatCompletionAssistantMessageParam{
			ToolCalls: []openai.ChatCompletionMessageToolCallParam{
				{
					ID: "call-" + name,
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      name,
						Arguments: `{}`,
					},
				},
			},
		},
	}
}
