package extproc

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/tidwall/sjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// jsonSchemaProbeFormat mirrors the reproduction in issue #3024: a strict
// json_schema whose single required property is distinctive enough to detect
// silent schema loss.
const jsonSchemaProbeFormat = `{
	"type": "json_schema",
	"json_schema": {
		"name": "probe",
		"strict": true,
		"schema": {
			"type": "object",
			"properties": {"zqx_answer": {"type": "string"}},
			"required": ["zqx_answer"],
			"additionalProperties": false
		}
	}
}`

func requestBodyWithResponseFormat(t *testing.T, format string) []byte {
	t.Helper()
	body := map[string]json.RawMessage{
		"model":    json.RawMessage(`"MoM"`),
		"messages": json.RawMessage(`[{"role":"user","content":"Write one sentence about mountains."}]`),
	}
	if format != "" {
		body["response_format"] = json.RawMessage(format)
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request body: %v", err)
	}
	return encoded
}

func decodedResponseFormat(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var decoded struct {
		ResponseFormat map[string]any `json:"response_format"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	return decoded.ResponseFormat
}

func expectedResponseFormat(t *testing.T, format string) map[string]any {
	t.Helper()
	var expected map[string]any
	if err := json.Unmarshal([]byte(format), &expected); err != nil {
		t.Fatalf("decode expected response_format: %v", err)
	}
	return expected
}

func TestParseOpenAIRequestRestoresJSONSchemaResponseFormat(t *testing.T) {
	body := requestBodyWithResponseFormat(t, jsonSchemaProbeFormat)

	req, err := parseOpenAIRequest(body)
	if err != nil {
		t.Fatalf("parseOpenAIRequest: %v", err)
	}
	if req.ResponseFormat.OfJSONSchema == nil {
		t.Fatalf("expected OfJSONSchema variant, got %+v", req.ResponseFormat)
	}
	if got := req.ResponseFormat.OfJSONSchema.JSONSchema.Name; got != "probe" {
		t.Fatalf("json_schema name = %q, want probe", got)
	}

	serialized, err := serializeOpenAIRequestWithStream(req, false)
	if err != nil {
		t.Fatalf("serializeOpenAIRequestWithStream: %v", err)
	}
	got := decodedResponseFormat(t, serialized)
	want := expectedResponseFormat(t, jsonSchemaProbeFormat)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("response_format lost through SDK round-trip:\n got %#v\nwant %#v", got, want)
	}
}

func TestParseOpenAIRequestResponseFormatVariants(t *testing.T) {
	t.Run("json_object", func(t *testing.T) {
		req, err := parseOpenAIRequest(requestBodyWithResponseFormat(t, `{"type":"json_object"}`))
		if err != nil {
			t.Fatalf("parseOpenAIRequest: %v", err)
		}
		if req.ResponseFormat.OfJSONObject == nil {
			t.Fatalf("expected OfJSONObject variant, got %+v", req.ResponseFormat)
		}
		serialized, err := serializeOpenAIRequestWithStream(req, false)
		if err != nil {
			t.Fatalf("serialize: %v", err)
		}
		if got := decodedResponseFormat(t, serialized); got["type"] != "json_object" {
			t.Fatalf("response_format = %#v, want json_object", got)
		}
	})

	t.Run("text", func(t *testing.T) {
		req, err := parseOpenAIRequest(requestBodyWithResponseFormat(t, `{"type":"text"}`))
		if err != nil {
			t.Fatalf("parseOpenAIRequest: %v", err)
		}
		if req.ResponseFormat.OfText == nil {
			t.Fatalf("expected OfText variant, got %+v", req.ResponseFormat)
		}
	})

	t.Run("payload_free_json_schema_stays_type_only", func(t *testing.T) {
		req, err := parseOpenAIRequest(requestBodyWithResponseFormat(t, `{"type":"json_schema"}`))
		if err != nil {
			t.Fatalf("parseOpenAIRequest: %v", err)
		}
		serialized, err := serializeOpenAIRequestWithStream(req, false)
		if err != nil {
			t.Fatalf("serialize: %v", err)
		}
		got := decodedResponseFormat(t, serialized)
		if !reflect.DeepEqual(got, map[string]any{"type": "json_schema"}) {
			t.Fatalf("payload-free json_schema mutated: %#v", got)
		}
	})

	t.Run("absent_response_format_untouched", func(t *testing.T) {
		req, err := parseOpenAIRequest(requestBodyWithResponseFormat(t, ""))
		if err != nil {
			t.Fatalf("parseOpenAIRequest: %v", err)
		}
		serialized, err := serializeOpenAIRequestWithStream(req, false)
		if err != nil {
			t.Fatalf("serialize: %v", err)
		}
		if got := decodedResponseFormat(t, serialized); got != nil {
			t.Fatalf("unexpected response_format: %#v", got)
		}
	})

	t.Run("unknown_type_left_to_sdk_behavior", func(t *testing.T) {
		req, err := parseOpenAIRequest(requestBodyWithResponseFormat(t, `{"type":"grammar","grammar":"root ::= \"x\""}`))
		if err != nil {
			t.Fatalf("parseOpenAIRequest: %v", err)
		}
		if req.ResponseFormat.OfJSONSchema != nil || req.ResponseFormat.OfJSONObject != nil {
			t.Fatalf("unknown type must not be coerced, got %+v", req.ResponseFormat)
		}
	})
}

func TestParseOpenAIRequestRejectsNonObjectJSONSchemaPayload(t *testing.T) {
	body := requestBodyWithResponseFormat(t, `{"type":"json_schema","json_schema":"not-an-object"}`)
	if _, err := parseOpenAIRequest(body); err == nil {
		t.Fatal("expected error for non-object json_schema payload, got nil")
	}
}

// TestRequestMutationsPreserveResponseFormat pins every request mutation on
// the auto-routing path as lossless for a strict json_schema response_format:
// model rewrite, stream field injection, reasoning-mode mutation, system
// prompt injection, request_params strip_unknown, and memory injection.
func TestRequestMutationsPreserveResponseFormat(t *testing.T) {
	body := requestBodyWithResponseFormat(t, jsonSchemaProbeFormat)
	want := expectedResponseFormat(t, jsonSchemaProbeFormat)

	assertFormatPreserved := func(t *testing.T, mutated []byte, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("mutation returned error: %v", err)
		}
		got := decodedResponseFormat(t, mutated)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("mutation dropped response_format:\n got %#v\nwant %#v", got, want)
		}
	}

	t.Run("model_rewrite", func(t *testing.T) {
		mutated, err := rewriteModelInBody(body, "qwen14b-dev")
		assertFormatPreserved(t, mutated, err)
	})

	t.Run("stream_fields", func(t *testing.T) {
		assertFormatPreserved(t, addStreamFieldsFast(body), nil)
	})

	t.Run("reasoning_mode", func(t *testing.T) {
		cfg := &config.RouterConfig{
			BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{
					"MoM": {ReasoningFamily: "qwen3"},
				},
			},
			IntelligentRouting: config.IntelligentRouting{
				ReasoningConfig: config.ReasoningConfig{
					ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
						"qwen3": {Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking"},
					},
				},
			},
		}
		router := &OpenAIRouter{Config: cfg}
		mutated, err := router.setReasoningModeToRequestBody(body, true, nil)
		assertFormatPreserved(t, mutated, err)
	})

	t.Run("system_prompt", func(t *testing.T) {
		mutated, _, err := addSystemPromptToRequestBody(body, "You answer with JSON.", "replace")
		assertFormatPreserved(t, mutated, err)
	})

	t.Run("request_params_strip_unknown", func(t *testing.T) {
		payload, err := config.NewStructuredPayload(map[string]interface{}{
			"strip_unknown": true,
		})
		if err != nil {
			t.Fatal(err)
		}
		decision := &config.Decision{
			Name:    "tier_a",
			Plugins: []config.DecisionPlugin{{Type: "request_params", Configuration: payload}},
		}
		withUnknown, err := sjson.SetBytes(body, "unknown_field", "x")
		if err != nil {
			t.Fatal(err)
		}
		mutated, err := (&OpenAIRouter{}).buildRequestParamsMutations(decision, withUnknown, nil, config.DefaultRecipeName)
		assertFormatPreserved(t, mutated, err)
		var decoded map[string]any
		if err := json.Unmarshal(mutated, &decoded); err != nil {
			t.Fatal(err)
		}
		if _, exists := decoded["unknown_field"]; exists {
			t.Fatal("strip_unknown did not run")
		}
	})

	t.Run("memory_injection", func(t *testing.T) {
		mutated, err := injectMemoryMessages(body, "remembered context")
		assertFormatPreserved(t, mutated, err)
	})
}

// TestModifyRequestBodyForAutoRoutingSerializePathKeepsJSONSchema covers the
// struct-serialization fallback (no raw working body), which is the shape the
// v0.3.0 auto-routing path always used and which modality, looper, and
// Anthropic-inbound flows still use.
func TestModifyRequestBodyForAutoRoutingSerializePathKeepsJSONSchema(t *testing.T) {
	body := requestBodyWithResponseFormat(t, jsonSchemaProbeFormat)
	req, err := parseOpenAIRequest(body)
	if err != nil {
		t.Fatalf("parseOpenAIRequest: %v", err)
	}

	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: map[string]string{}}

	modified, err := router.modifyRequestBodyForAutoRouting(req, "qwen14b-dev", "", false, nil, ctx)
	if err != nil {
		t.Fatalf("modifyRequestBodyForAutoRouting: %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(modified, &decoded); err != nil {
		t.Fatalf("decode modified body: %v", err)
	}
	if got := decoded["model"]; got != "qwen14b-dev" {
		t.Fatalf("model = %#v, want qwen14b-dev", got)
	}
	got := decodedResponseFormat(t, modified)
	want := expectedResponseFormat(t, jsonSchemaProbeFormat)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("serialize fallback dropped response_format:\n got %#v\nwant %#v", got, want)
	}
}

// TestHandleAutoModelRoutingPreservesJSONSchemaResponseFormat mirrors issue
// #3024 end to end at the routing layer: a request naming the auto model with
// a strict json_schema response_format must reach the backend with the full
// schema payload after the model rewrite.
func TestHandleAutoModelRoutingPreservesJSONSchemaResponseFormat(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "qwen14b-dev",
			ModelConfig: map[string]config.ModelParams{
				"qwen14b-dev": {PreferredEndpoints: []string{"qwen14b-dev_vllm"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name:    "qwen14b-dev_vllm",
				Address: "127.0.0.1",
				Port:    8000,
				Type:    "vllm",
				Weight:  1,
			}},
		},
	}
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
	originalBody := requestBodyWithResponseFormat(t, jsonSchemaProbeFormat)
	ctx := &RequestContext{
		Headers:             map[string]string{},
		TraceContext:        context.Background(),
		OriginalRequestBody: originalBody,
	}
	req, err := parseOpenAIRequest(originalBody)
	if err != nil {
		t.Fatalf("parseOpenAIRequest: %v", err)
	}
	baseResponse := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{Status: ext_proc.CommonResponse_CONTINUE},
			},
		},
	}

	response, err := router.handleAutoModelRouting(
		req,
		"MoM",
		"",
		entropy.ReasoningDecision{},
		"qwen14b-dev",
		ctx,
		baseResponse,
	)
	if err != nil {
		t.Fatalf("handleAutoModelRouting: %v", err)
	}

	mutated := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	if len(mutated) == 0 {
		t.Fatal("expected body mutation")
	}
	var decoded map[string]any
	if err := json.Unmarshal(mutated, &decoded); err != nil {
		t.Fatalf("decode mutated body: %v", err)
	}
	if got := decoded["model"]; got != "qwen14b-dev" {
		t.Fatalf("model = %#v, want qwen14b-dev", got)
	}
	got := decodedResponseFormat(t, mutated)
	want := expectedResponseFormat(t, jsonSchemaProbeFormat)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("auto routing dropped response_format:\n got %#v\nwant %#v", got, want)
	}
}
