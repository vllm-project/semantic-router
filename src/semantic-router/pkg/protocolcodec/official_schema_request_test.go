package protocolcodec

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

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

func TestRemovedResponsesItemCacheBreakpointFailsClosed(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"model":"m","input":[{"type":"function_call","id":"item_1","call_id":"call_1","name":"lookup","arguments":"{}","prompt_cache_breakpoint":{"mode":"explicit"}}]}`)
	_, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != llmprotocol.ErrorInvalidRequest {
		t.Fatalf("removed Responses item cache breakpoint returned %T %v, want typed invalid_request", err, err)
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
