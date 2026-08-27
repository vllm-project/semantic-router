package protocolcodec

import (
	"encoding/json"
	"errors"
	"reflect"
	"sort"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// These inventories are the top-level request fields published by the
// OpenAI OpenAPI contract at 172101000e7be21103c405aa8bedf918039f886f and the
// generated Anthropic Messages API types at
// f6f796100d7bb958d84580f44060a0a2b21bfe04.
// Every field is either represented semantically or decoded into an explicit
// unsupported_feature error; adding a silent JSON sink is not allowed.
func TestOfficialRequestFieldInventoriesAreClosed(t *testing.T) {
	tests := []struct {
		name       string
		wire       any
		official   []string
		extensions []string
	}{
		{
			name: "OpenAI Chat Completions",
			wire: chatRequestWire{},
			official: fields(
				"audio", "frequency_penalty", "function_call", "functions", "logit_bias", "logprobs",
				"max_completion_tokens", "max_tokens", "messages", "metadata", "modalities", "model",
				"moderation", "n", "parallel_tool_calls", "prediction", "presence_penalty",
				"prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning_effort",
				"response_format", "safety_identifier", "seed", "service_tier", "stop", "store", "stream",
				"stream_options", "temperature", "tool_choice", "tools", "top_logprobs", "top_p", "user",
				"verbosity", "web_search_options",
			),
			extensions: fields("reasoning_budget_tokens"),
		},
		{
			name: "OpenAI Responses",
			wire: responsesRequestWire{},
			official: fields(
				"background", "context_management", "conversation", "include", "input", "instructions",
				"max_output_tokens", "max_tool_calls", "metadata", "model", "moderation", "parallel_tool_calls",
				"previous_response_id", "prompt", "prompt_cache_key", "prompt_cache_options",
				"prompt_cache_retention", "reasoning", "safety_identifier", "service_tier", "store", "stream",
				"stream_options", "temperature", "text", "tool_choice", "tools", "top_logprobs", "top_p",
				"truncation", "user",
			),
			extensions: fields("auto_store"),
		},
		{
			name: "Anthropic Messages",
			wire: anthropicRequestWire{},
			official: fields(
				"cache_control", "container", "inference_geo", "max_tokens", "messages", "metadata", "model",
				"output_config", "service_tier", "stop_sequences", "stream", "system", "temperature", "thinking",
				"tool_choice", "tools", "top_k", "top_p",
			),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			want := append(append([]string(nil), test.official...), test.extensions...)
			sort.Strings(want)
			if got := jsonFieldNames(reflect.TypeOf(test.wire)); !reflect.DeepEqual(got, want) {
				t.Fatalf("wire field inventory drifted\n got: %v\nwant: %v", got, want)
			}
		})
	}
}

func TestOfficialResponseFieldInventoriesAreClosed(t *testing.T) {
	tests := []struct {
		name       string
		wire       any
		official   []string
		extensions []string
	}{
		{
			name: "OpenAI Chat Completions",
			wire: chatResponseWire{},
			official: fields(
				"choices", "created", "id", "metadata", "model", "moderation", "object", "service_tier", "system_fingerprint", "usage",
			),
			extensions: fields(
				"ec_transfer_params", "error", "kv_transfer_params", "metrics", "prompt_logprobs",
				"prompt_text", "prompt_token_ids",
			),
		},
		{
			name: "OpenAI Responses",
			wire: responsesResponseWire{},
			official: fields(
				"background", "completed_at", "conversation", "created_at", "error", "id",
				"incomplete_details", "instructions", "max_output_tokens", "max_tool_calls", "metadata",
				"model", "moderation", "object", "output", "output_text", "parallel_tool_calls", "previous_response_id",
				"prompt", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning",
				"safety_identifier", "service_tier", "status", "store", "temperature", "text", "tool_choice",
				"tools", "top_logprobs", "top_p", "truncation", "usage", "user",
			),
			extensions: fields("conversation_id"),
		},
		{
			name: "Anthropic Messages",
			wire: anthropicResponseWire{},
			official: fields(
				"container", "content", "id", "model", "role", "stop_details", "stop_reason",
				"stop_sequence", "type", "usage",
			),
			extensions: fields("error"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			want := append(append([]string(nil), test.official...), test.extensions...)
			sort.Strings(want)
			if got := jsonFieldNames(reflect.TypeOf(test.wire)); !reflect.DeepEqual(got, want) {
				t.Fatalf("wire field inventory drifted\n got: %v\nwant: %v", got, want)
			}
		})
	}
}

func TestOfficialUsageFieldInventoriesAreClosed(t *testing.T) {
	tests := []struct {
		name     string
		wire     any
		official []string
	}{
		{
			name: "OpenAI Chat Completions",
			wire: chatUsageWire{},
			official: fields(
				"completion_tokens", "completion_tokens_details", "compute_units", "prompt_tokens",
				"prompt_tokens_details", "total_tokens",
			),
		},
		{
			name: "OpenAI Responses",
			wire: responsesUsageWire{},
			official: fields(
				"compute_units", "input_tokens", "input_tokens_details", "output_tokens",
				"output_tokens_details", "total_tokens",
			),
		},
		{
			name: "Anthropic Messages",
			wire: anthropicUsageWire{},
			official: fields(
				"cache_creation", "cache_creation_input_tokens", "cache_read_input_tokens", "inference_geo",
				"input_tokens", "output_tokens", "output_tokens_details", "server_tool_use", "service_tier",
			),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := jsonFieldNames(reflect.TypeOf(test.wire)); !reflect.DeepEqual(got, test.official) {
				t.Fatalf("usage field inventory drifted\n got: %v\nwant: %v", got, test.official)
			}
		})
	}
}

func TestOfficialNestedFieldInventoriesAreClosed(t *testing.T) {
	tests := []struct {
		name       string
		wire       any
		official   []string
		extensions []string
	}{
		{
			name: "OpenAI Chat message union", wire: chatMessageWire{},
			official:   fields("audio", "content", "function_call", "name", "refusal", "role", "tool_call_id", "tool_calls"),
			extensions: fields("annotations", "id", "reasoning", "reasoning_content"),
		},
		{
			name: "OpenAI Chat content-part union", wire: chatContentWire{},
			official:   fields("file", "image_url", "input_audio", "prompt_cache_breakpoint", "text", "type"),
			extensions: fields("refusal"),
		},
		{
			name: "OpenAI Chat file input", wire: chatFileWire{},
			official: fields("file_data", "file_id", "filename"),
		},
		{
			name: "OpenAI Chat function definition", wire: chatFunctionDefinitionWire{},
			official: fields("description", "name", "parameters", "strict"),
		},
		{
			name: "OpenAI Chat function call", wire: chatFunctionCallWire{},
			official: fields("arguments", "name"),
		},
		{
			name: "OpenAI Responses function tool", wire: responsesToolWire{},
			official: fields("allowed_callers", "defer_loading", "description", "name", "output_schema", "parameters", "strict", "type"),
		},
		{
			name: "OpenAI Responses content-part union", wire: responsesContentWire{},
			official: fields(
				"annotations", "detail", "file_data", "file_id", "file_url", "filename", "image_url",
				"logprobs", "prompt_cache_breakpoint", "refusal", "text", "type",
			),
		},
		{
			name: "OpenAI Responses supported item union", wire: responsesItemWire{},
			official: fields(
				"arguments", "call_id", "content", "encrypted_content", "id", "name", "output", "phase",
				"role", "status", "summary", "type",
			),
		},
		{
			name: "Anthropic message", wire: anthropicMessageWire{},
			official: fields("content", "role"),
		},
		{
			name: "Anthropic supported content-block union", wire: anthropicContentWire{},
			official: fields(
				"cache_control", "caller", "citations", "content", "context", "id", "input", "is_error",
				"name", "signature", "source", "text", "thinking", "title", "tool_use_id", "toolset_name",
				"transformations", "type",
			),
		},
		{
			name: "Anthropic media-source union", wire: anthropicMediaSourceWire{},
			official: fields("content", "data", "file_id", "media_type", "type", "url"),
		},
		{
			name: "Anthropic custom tool", wire: anthropicToolWire{},
			official: fields(
				"allowed_callers", "cache_control", "defer_loading", "description", "eager_input_streaming",
				"input_examples", "input_schema", "name", "strict", "type",
			),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			want := append(append([]string(nil), test.official...), test.extensions...)
			sort.Strings(want)
			if got := jsonFieldNames(reflect.TypeOf(test.wire)); !reflect.DeepEqual(got, want) {
				t.Fatalf("nested wire field inventory drifted\n got: %v\nwant: %v", got, want)
			}
		})
	}
}

func TestOfficialNestedUnsupportedFieldsFailWithTypedErrors(t *testing.T) {
	tests := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
	}{
		{"Chat message name", llmprotocol.OpenAIChatV1, `{"model":"m","messages":[{"role":"user","name":"alice","content":"hello"}]}`},
		{"Chat cache breakpoint", llmprotocol.OpenAIChatV1, `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hello","prompt_cache_breakpoint":{"type":"ephemeral"}}]}]}`},
		{"Responses deferred tool", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":"hello","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"},"defer_loading":true}]}`},
		{"Responses input breakpoint", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello","prompt_cache_breakpoint":{"type":"ephemeral"}}]}]}`},
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

func TestOfficialUnsupportedResponsesItemDiscriminatorsAreTyped(t *testing.T) {
	unsupported := fields(
		"additional_tools", "apply_patch_call", "apply_patch_call_output", "code_interpreter_call",
		"compaction", "compaction_trigger", "computer_call", "computer_call_output", "custom_tool_call",
		"custom_tool_call_output", "file_search_call", "function_shell_call",
		"function_shell_call_output", "image_generation_call", "local_shell_call", "local_shell_call_output",
		"mcp_approval_request", "mcp_approval_response", "mcp_call", "mcp_list_tools", "program",
		"program_output", "tool_search_call", "tool_search_output", "web_search_call",
	)
	engine := NewBuiltinEngine()
	for _, itemType := range unsupported {
		t.Run(itemType, func(t *testing.T) {
			body, err := json.Marshal(map[string]any{
				"model": "m",
				"input": []map[string]any{{"type": itemType, "id": "item_1"}},
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
				"service_tier", "stream_options", "top_logprobs",
			),
		},
		{
			format: llmprotocol.AnthropicMessagesV1,
			base: map[string]any{
				"model": "m", "max_tokens": 16,
				"messages": []any{map[string]any{"role": "user", "content": "hello"}},
			},
			fields: fields("cache_control", "container", "inference_geo", "output_config", "service_tier"),
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

func TestOfficialIdentityStateAndReasoningFieldsTranslateSemantically(t *testing.T) {
	engine := NewBuiltinEngine()
	assertOfficialChatIdentityState(t, engine)
	assertOfficialResponsesIdentityState(t, engine)
	assertOfficialAnthropicIdentityState(t, engine)
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
	responsesBody := []byte(`{
		"id":"resp_1","object":"response","created_at":1,"completed_at":2,"status":"completed",
		"error":null,"incomplete_details":null,"instructions":"answer","max_output_tokens":64,
		"model":"m","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"done"}]}],
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

	anthropicBody := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-test","container":{},
		"content":[{"type":"text","text":"done"}],"stop_details":null,"stop_reason":"end_turn","stop_sequence":null,
		"usage":{"cache_creation":{"ephemeral_1h_input_tokens":1,"ephemeral_5m_input_tokens":2},
		"cache_creation_input_tokens":3,"cache_read_input_tokens":4,"inference_geo":"us","input_tokens":10,
		"output_tokens":6,"output_tokens_details":{"thinking_tokens":2},"server_tool_use":{"web_search_requests":1},
		"service_tier":"standard"}
	}`)
	response, _, diagnostics, err = engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, anthropicBody)
	if err != nil {
		t.Fatal(err)
	}
	if tokenValue(response.Usage.OutputReasoning) != 2 || tokenValue(response.Usage.OutputOther) != 4 || len(diagnostics) == 0 {
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
		"output":[{"type":"message","id":"msg_1","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"done","logprobs":[]}]}],
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
		"prompt", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention",
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

func fields(values ...string) []string {
	result := append([]string(nil), values...)
	sort.Strings(result)
	return result
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
