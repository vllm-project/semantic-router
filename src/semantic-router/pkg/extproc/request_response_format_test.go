package extproc

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// jsonSchemaProbeFormat mirrors issue #3024. A type-only response_format is
// still valid JSON, so the nested schema must be compared semantically.
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

func parseRequestWithFormat(t *testing.T, format string) *openai.ChatCompletionNewParams {
	t.Helper()
	req, err := parseOpenAIRequest(requestBodyWithResponseFormat(t, format))
	if err != nil {
		t.Fatalf("parseOpenAIRequest: %v", err)
	}
	return req
}

func serializeSDKRequest(t *testing.T, req *openai.ChatCompletionNewParams) []byte {
	t.Helper()
	serialized, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal SDK request: %v", err)
	}
	return serialized
}

func TestParseOpenAIRequestRestoresJSONSchemaResponseFormat(t *testing.T) {
	req := parseRequestWithFormat(t, jsonSchemaProbeFormat)
	if req.ResponseFormat.OfJSONSchema == nil {
		t.Fatalf("expected OfJSONSchema variant, got %+v", req.ResponseFormat)
	}
	if got := req.ResponseFormat.OfJSONSchema.JSONSchema.Name; got != "probe" {
		t.Fatalf("json_schema name = %q, want probe", got)
	}
	got := decodedResponseFormat(t, serializeSDKRequest(t, req))
	want := expectedResponseFormat(t, jsonSchemaProbeFormat)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("response_format lost through SDK round trip:\n got %#v\nwant %#v", got, want)
	}
}

func TestParseOpenAIRequestResponseFormatVariants(t *testing.T) {
	tests := []struct {
		name   string
		format string
		kind   string
	}{
		{name: "json_object", format: `{"type":"json_object"}`, kind: "json_object"},
		{name: "text", format: `{"type":"text"}`, kind: "text"},
		{name: "json_schema_without_payload", format: `{"type":"json_schema"}`, kind: "json_schema"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := decodedResponseFormat(t, serializeSDKRequest(t, parseRequestWithFormat(t, test.format)))
			if got["type"] != test.kind {
				t.Fatalf("response_format = %#v, want type %q", got, test.kind)
			}
		})
	}

	if got := decodedResponseFormat(t, serializeSDKRequest(t, parseRequestWithFormat(t, ""))); got != nil {
		t.Fatalf("unexpected response_format: %#v", got)
	}
}

func TestParseOpenAIRequestRejectsNonObjectJSONSchemaPayload(t *testing.T) {
	body := requestBodyWithResponseFormat(t, `{"type":"json_schema","json_schema":"not-an-object"}`)
	if _, err := parseOpenAIRequest(body); err == nil {
		t.Fatal("expected error for non-object json_schema payload, got nil")
	}
}

// Looper and internal model calls receive a neutral request encoded as Chat
// Completions and then parsed into the provider SDK type. Pin that production
// bridge independently from the exhaustive 3x3 ExtProc matrix.
func TestNeutralRequestSDKBridgePreservesJSONSchemaResponseFormat(t *testing.T) {
	engine := protocolcodec.NewBuiltinEngine()
	body := requestBodyWithResponseFormat(t, jsonSchemaProbeFormat)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "routed-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	sdkRequest, err := parseOpenAIRequest(encoded.Body)
	if err != nil {
		t.Fatal(err)
	}
	got := decodedResponseFormat(t, serializeSDKRequest(t, sdkRequest))
	want := expectedResponseFormat(t, jsonSchemaProbeFormat)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("neutral-to-SDK bridge changed response_format:\n got %#v\nwant %#v", got, want)
	}
}
