package protocolcodec

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenAIChatMetadataNonStringValues(t *testing.T) {
	codec := OpenAIChatCodec{}
	policy := llmprotocol.DefaultPolicy()

	cases := []struct {
		name     string
		metadata string
		key      string
		wantVal  string
	}{
		{
			name:     "string value",
			metadata: `{"a":"b"}`,
			key:      "a",
			wantVal:  "b",
		},
		{
			name:     "list of strings (LiteLLM tags)",
			metadata: `{"tags":["agent","prod"]}`,
			key:      "tags",
			wantVal:  `["agent","prod"]`,
		},
		{
			name:     "integer number",
			metadata: `{"turn":1}`,
			key:      "turn",
			wantVal:  "1",
		},
		{
			name:     "floating point number",
			metadata: `{"score":98.5}`,
			key:      "score",
			wantVal:  "98.5",
		},
		{
			name:     "boolean true",
			metadata: `{"flag":true}`,
			key:      "flag",
			wantVal:  "true",
		},
		{
			name:     "boolean false",
			metadata: `{"flag":false}`,
			key:      "flag",
			wantVal:  "false",
		},
		{
			name:     "nested object",
			metadata: `{"nested":{"key":"val"}}`,
			key:      "nested",
			wantVal:  `{"key":"val"}`,
		},
		{
			name:     "null value",
			metadata: `{"empty":null}`,
			key:      "empty",
			wantVal:  "null",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rawBody := []byte(`{"model":"vllm-sr/auto","stream":true,"messages":[{"role":"user","content":"hi"}],"metadata":` + tc.metadata + `}`)
			req, envelope, _, err := codec.DecodeRequest(rawBody, policy)
			if err != nil {
				t.Fatalf("unexpected error decoding metadata %s: %v", tc.metadata, err)
			}
			if req.Metadata == nil {
				t.Fatalf("req.Metadata is nil")
			}
			if got := req.Metadata[tc.key]; got != tc.wantVal {
				t.Errorf("Metadata[%q] = %q, want %q", tc.key, got, tc.wantVal)
			}
			if req.MetadataJSON == nil || len(req.MetadataJSON[tc.key]) == 0 {
				t.Fatalf("req.MetadataJSON missing key %q", tc.key)
			}
			if err = llmprotocol.ValidateRequest(req, policy.Limits); err != nil {
				t.Errorf("ValidateRequest failed: %v", err)
			}

			// Same-format re-encoding should replay verbatim bytes via envelope
			var encoded []byte
			encoded, _, err = codec.EncodeRequest(req, envelope, policy)
			if err != nil {
				t.Fatalf("EncodeRequest replay failed: %v", err)
			}
			if !bytes.Equal(encoded, rawBody) {
				t.Errorf("replayed bytes mismatch: got %s, want %s", string(encoded), string(rawBody))
			}

			// Forced re-encoding (empty envelope, e.g. when router adds include_usage)
			// should preserve original JSON types rather than stringifying them.
			var reencoded []byte
			reencoded, _, err = codec.EncodeRequest(req, llmprotocol.Envelope{}, policy)
			if err != nil {
				t.Fatalf("Forced EncodeRequest failed: %v", err)
			}
			var parsed map[string]any
			if err := json.Unmarshal(reencoded, &parsed); err != nil {
				t.Fatalf("unmarshal reencoded JSON: %v", err)
			}
			meta, ok := parsed["metadata"].(map[string]any)
			if !ok {
				t.Fatalf("re-encoded metadata is not a map: %v", parsed["metadata"])
			}
			val := meta[tc.key]
			switch tc.name {
			case "string value":
				if val != "b" {
					t.Errorf("reencoded string: got %v (%T), want 'b'", val, val)
				}
			case "list of strings (LiteLLM tags)":
				list, ok := val.([]any)
				if !ok || len(list) != 2 || list[0] != "agent" || list[1] != "prod" {
					t.Errorf("reencoded list: got %v (%T), want []any{'agent', 'prod'}", val, val)
				}
			case "integer number":
				if val != float64(1) {
					t.Errorf("reencoded int: got %v (%T), want 1", val, val)
				}
			case "floating point number":
				if val != 98.5 {
					t.Errorf("reencoded float: got %v (%T), want 98.5", val, val)
				}
			case "boolean true":
				if val != true {
					t.Errorf("reencoded bool: got %v (%T), want true", val, val)
				}
			case "boolean false":
				if val != false {
					t.Errorf("reencoded bool: got %v (%T), want false", val, val)
				}
			case "nested object":
				obj, ok := val.(map[string]any)
				if !ok || obj["key"] != "val" {
					t.Errorf("reencoded object: got %v (%T), want map with key=val", val, val)
				}
			case "null value":
				if val != nil {
					t.Errorf("reencoded null: got %v (%T), want nil", val, val)
				}
			}
		})
	}
}

func TestOpenAIMetadataLargeIntegerPrecisionPreserved(t *testing.T) {
	codec := OpenAIChatCodec{}
	policy := llmprotocol.DefaultPolicy()

	// 9007199254740993 is 2^53 + 1, which loses precision in float64 (truncating to 9007199254740992)
	raw := []byte(`{"model":"vllm-sr/auto","stream":true,"messages":[{"role":"user","content":"hi"}],"metadata":{"large_int":9007199254740993}}`)
	req, _, _, err := codec.DecodeRequest(raw, policy)
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}

	if req.Metadata["large_int"] != "9007199254740993" {
		t.Errorf("req.Metadata[large_int] = %q, want 9007199254740993", req.Metadata["large_int"])
	}
	if string(req.MetadataJSON["large_int"]) != "9007199254740993" {
		t.Errorf("req.MetadataJSON[large_int] = %s, want 9007199254740993", string(req.MetadataJSON["large_int"]))
	}

	// Forced re-encoding (e.g. streaming Chat where router adds include_usage)
	reencoded, _, err := codec.EncodeRequest(req, llmprotocol.Envelope{}, policy)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	if !bytes.Contains(reencoded, []byte(`"large_int":9007199254740993`)) {
		t.Errorf("re-encoded wire JSON lost large integer precision: %s", string(reencoded))
	}
}

func TestOpenAIChatCompoundMetadata(t *testing.T) {
	codec := OpenAIChatCodec{}
	policy := llmprotocol.DefaultPolicy()

	raw := []byte(`{
		"model":"vllm-sr/auto",
		"messages":[{"role":"user","content":"hi"}],
		"metadata":{
			"trace_id":"trace-123",
			"session_id":"sess-456",
			"tags":["agent","vllm"],
			"turn":3,
			"score":99.25,
			"active":true,
			"extra":{"debug":true}
		}
	}`)

	req, _, _, err := codec.DecodeRequest(raw, policy)
	if err != nil {
		t.Fatalf("decode compound metadata: %v", err)
	}

	expected := map[string]string{
		"trace_id":   "trace-123",
		"session_id": "sess-456",
		"tags":       `["agent","vllm"]`,
		"turn":       "3",
		"score":      "99.25",
		"active":     "true",
		"extra":      `{"debug":true}`,
	}

	for k, want := range expected {
		if got := req.Metadata[k]; got != want {
			t.Errorf("Metadata[%q] = %q, want %q", k, got, want)
		}
	}

	// Verify forced re-encode retains compound JSON structure
	reencoded, _, err := codec.EncodeRequest(req, llmprotocol.Envelope{}, policy)
	if err != nil {
		t.Fatalf("forced encode compound metadata: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(reencoded, &parsed); err != nil {
		t.Fatalf("unmarshal reencoded JSON: %v", err)
	}
	meta := parsed["metadata"].(map[string]any)
	if meta["trace_id"] != "trace-123" {
		t.Errorf("reencoded trace_id = %v", meta["trace_id"])
	}
	if meta["turn"] != float64(3) {
		t.Errorf("reencoded turn = %v", meta["turn"])
	}
	if tags, ok := meta["tags"].([]any); !ok || len(tags) != 2 {
		t.Errorf("reencoded tags = %v", meta["tags"])
	}
	if extra, ok := meta["extra"].(map[string]any); !ok || extra["debug"] != true {
		t.Errorf("reencoded extra = %v", meta["extra"])
	}
}

func TestOpenAIResponsesMetadataNonStringValues(t *testing.T) {
	codec := OpenAIResponsesCodec{}
	policy := llmprotocol.DefaultPolicy()

	raw := []byte(`{
		"model":"vllm-sr/auto",
		"input":"hi",
		"metadata":{
			"trace_id":"trace-123",
			"tags":["agent","prod"],
			"turn":2,
			"enabled":true,
			"config":{"retry":3}
		},
		"client_metadata":{
			"client_tags":["telemetry"],
			"client_turn":1,
			"client_flag":false
		}
	}`)

	req, envelope, diagnostics, err := codec.DecodeRequest(raw, policy)
	if err != nil {
		t.Fatalf("decode responses request: %v", err)
	}

	if req.Metadata["trace_id"] != "trace-123" {
		t.Errorf("Metadata[trace_id] = %q, want trace-123", req.Metadata["trace_id"])
	}
	if req.Metadata["tags"] != `["agent","prod"]` {
		t.Errorf("Metadata[tags] = %q, want [\"agent\",\"prod\"]", req.Metadata["tags"])
	}
	if req.Metadata["turn"] != "2" {
		t.Errorf("Metadata[turn] = %q, want 2", req.Metadata["turn"])
	}
	if req.Metadata["enabled"] != "true" {
		t.Errorf("Metadata[enabled] = %q, want true", req.Metadata["enabled"])
	}
	if req.Metadata["config"] != `{"retry":3}` {
		t.Errorf("Metadata[config] = %q, want {\"retry\":3}", req.Metadata["config"])
	}

	// Client metadata should be omitted with a diagnostic
	foundDiagnostic := false
	for _, diag := range diagnostics {
		if diag.Field == "client_metadata" {
			foundDiagnostic = true
			break
		}
	}
	if !foundDiagnostic {
		t.Errorf("expected client_metadata diagnostic omission, got: %v", diagnostics)
	}

	if err = llmprotocol.ValidateRequest(req, policy.Limits); err != nil {
		t.Errorf("ValidateRequest failed: %v", err)
	}

	// Replay preserves bytes
	var encoded []byte
	encoded, _, err = codec.EncodeRequest(req, envelope, policy)
	if err != nil {
		t.Fatalf("EncodeRequest failed: %v", err)
	}
	if !bytes.Equal(encoded, raw) {
		t.Errorf("replayed bytes mismatch")
	}

	// Forced re-encode preserves native JSON types
	reencoded, _, err := codec.EncodeRequest(req, llmprotocol.Envelope{}, policy)
	if err != nil {
		t.Fatalf("forced EncodeRequest failed: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(reencoded, &parsed); err != nil {
		t.Fatalf("unmarshal reencoded JSON: %v", err)
	}
	meta := parsed["metadata"].(map[string]any)
	if meta["trace_id"] != "trace-123" {
		t.Errorf("trace_id: got %v, want trace-123", meta["trace_id"])
	}
	if tags, ok := meta["tags"].([]any); !ok || len(tags) != 2 || tags[0] != "agent" {
		t.Errorf("tags: got %v, want ['agent', 'prod']", meta["tags"])
	}
	if meta["turn"] != float64(2) {
		t.Errorf("turn: got %v, want 2", meta["turn"])
	}
	if meta["enabled"] != true {
		t.Errorf("enabled: got %v, want true", meta["enabled"])
	}
	if cfg, ok := meta["config"].(map[string]any); !ok || cfg["retry"] != float64(3) {
		t.Errorf("config: got %v, want map[retry:3]", meta["config"])
	}
}

func TestAnthropicRejectsMetadataTranslation(t *testing.T) {
	chatCodec := OpenAIChatCodec{}
	anthropicCodec := AnthropicMessagesCodec{}
	policy := llmprotocol.DefaultPolicy()

	raw := []byte(`{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"hi"}],"metadata":{"tags":["agent"],"turn":1}}`)
	req, _, _, err := chatCodec.DecodeRequest(raw, policy)
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}

	// Attempting to cross-translate metadata to Anthropic Messages should be rejected
	_, _, err = anthropicCodec.EncodeRequest(req, llmprotocol.Envelope{}, policy)
	if err == nil {
		t.Fatalf("expected error translating metadata to Anthropic Messages, got nil")
	}

	protocolErr, ok := err.(*llmprotocol.ProtocolError)
	if !ok {
		t.Fatalf("expected *llmprotocol.ProtocolError, got %T: %v", err, err)
	}
	if protocolErr.Code != "unsupported_metadata" {
		t.Errorf("error code = %q, want unsupported_metadata", protocolErr.Code)
	}
	if protocolErr.Category != llmprotocol.ErrorUnsupportedFeature {
		t.Errorf("error category = %q, want %q", protocolErr.Category, llmprotocol.ErrorUnsupportedFeature)
	}
}

func TestOpenAIChatAndResponsesMetadataInvalidTypes(t *testing.T) {
	chatCodec := OpenAIChatCodec{}
	responsesCodec := OpenAIResponsesCodec{}
	policy := llmprotocol.DefaultPolicy()

	invalidPayloads := []struct {
		name string
		chat string
		resp string
	}{
		{
			name: "string instead of map",
			chat: `{"model":"m","messages":[{"role":"user","content":"hi"}],"metadata":"not-a-map"}`,
			resp: `{"model":"m","input":"hi","metadata":"not-a-map"}`,
		},
		{
			name: "array instead of map",
			chat: `{"model":"m","messages":[{"role":"user","content":"hi"}],"metadata":[1,2,3]}`,
			resp: `{"model":"m","input":"hi","metadata":[1,2,3]}`,
		},
		{
			name: "number instead of map",
			chat: `{"model":"m","messages":[{"role":"user","content":"hi"}],"metadata":123}`,
			resp: `{"model":"m","input":"hi","metadata":123}`,
		},
	}

	for _, tc := range invalidPayloads {
		t.Run("chat_"+tc.name, func(t *testing.T) {
			_, _, _, err := chatCodec.DecodeRequest([]byte(tc.chat), policy)
			if err == nil {
				t.Fatalf("expected error decoding invalid metadata type in chat request, got nil")
			}
		})
		t.Run("responses_"+tc.name, func(t *testing.T) {
			_, _, _, err := responsesCodec.DecodeRequest([]byte(tc.resp), policy)
			if err == nil {
				t.Fatalf("expected error decoding invalid metadata type in responses request, got nil")
			}
		})
	}
}

func TestMetadataJSONHelpers(t *testing.T) {
	if got := stringifyMetadataJSONValue(json.RawMessage(`"hello"`)); got != "hello" {
		t.Errorf("stringify string: got %q, want %q", got, "hello")
	}
	if got := stringifyMetadataJSONValue(json.RawMessage(`true`)); got != "true" {
		t.Errorf("stringify true: got %q, want %q", got, "true")
	}
	if got := stringifyMetadataJSONValue(json.RawMessage(`false`)); got != "false" {
		t.Errorf("stringify false: got %q, want %q", got, "false")
	}
	if got := stringifyMetadataJSONValue(json.RawMessage(`42`)); got != "42" {
		t.Errorf("stringify int: got %q, want %q", got, "42")
	}
	if got := stringifyMetadataJSONValue(json.RawMessage(`9007199254740993`)); got != "9007199254740993" {
		t.Errorf("stringify large int: got %q, want %q", got, "9007199254740993")
	}
	if got := stringifyMetadataJSONValue(json.RawMessage(`["a","b"]`)); got != `["a","b"]` {
		t.Errorf("stringify list: got %q, want [\"a\",\"b\"]", got)
	}
	if got := stringifyMetadataJSONValue(json.RawMessage(`null`)); got != "null" {
		t.Errorf("stringify null: got %q, want null", got)
	}
	if got := stringifyMetadataJSONValue(nil); got != "" {
		t.Errorf("stringify empty: got %q, want empty", got)
	}

	if got := stringifyWireMetadataJSON(nil); got != nil {
		t.Errorf("stringify nil map: got %v, want nil", got)
	}
	if got := cloneWireMetadataJSON(nil); got != nil {
		t.Errorf("clone nil map: got %v, want nil", got)
	}
	if got := encodeWireMetadataJSON(llmprotocol.Request{}); got != nil {
		t.Errorf("encode empty request: got %v, want nil", got)
	}

	reqWithStrings := llmprotocol.Request{
		Metadata: map[string]string{"key": "val"},
	}
	encoded := encodeWireMetadataJSON(reqWithStrings)
	if string(encoded["key"]) != `"val"` {
		t.Errorf("encode string-only map: got %s, want %q", string(encoded["key"]), `"val"`)
	}
}
