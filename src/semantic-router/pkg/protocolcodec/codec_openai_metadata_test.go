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
			if err = llmprotocol.ValidateRequest(req, policy.Limits); err != nil {
				t.Errorf("ValidateRequest failed: %v", err)
			}

			// Same-format re-encoding should replay verbatim bytes
			var encoded []byte
			encoded, _, err = codec.EncodeRequest(req, envelope, policy)
			if err != nil {
				t.Fatalf("EncodeRequest failed: %v", err)
			}
			if !bytes.Equal(encoded, rawBody) {
				t.Errorf("replayed bytes mismatch: got %s, want %s", string(encoded), string(rawBody))
			}

			// Forced re-encoding (empty envelope) should encode valid JSON with metadata
			var reencoded []byte
			reencoded, _, err = codec.EncodeRequest(req, llmprotocol.Envelope{}, policy)
			if err != nil {
				t.Fatalf("Forced EncodeRequest failed: %v", err)
			}
			var roundtrip chatRequestWire
			if err := json.Unmarshal(reencoded, &roundtrip); err != nil {
				t.Fatalf("unmarshal roundtrip wire: %v", err)
			}
			if roundtrip.Metadata == nil || roundtrip.Metadata[tc.key] == nil {
				t.Errorf("roundtrip metadata missing key %q: %v", tc.key, roundtrip.Metadata)
			}
		})
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

func TestMetadataCoercionHelpers(t *testing.T) {
	if got := coerceMetadataValue("hello"); got != "hello" {
		t.Errorf("coerce string: got %q, want %q", got, "hello")
	}
	if got := coerceMetadataValue(true); got != "true" {
		t.Errorf("coerce true: got %q, want %q", got, "true")
	}
	if got := coerceMetadataValue(false); got != "false" {
		t.Errorf("coerce false: got %q, want %q", got, "false")
	}
	if got := coerceMetadataValue(42); got != "42" {
		t.Errorf("coerce int: got %q, want %q", got, "42")
	}
	if got := coerceMetadataValue(int64(100)); got != "100" {
		t.Errorf("coerce int64: got %q, want %q", got, "100")
	}
	if got := coerceMetadataValue(float64(5)); got != "5" {
		t.Errorf("coerce float integer: got %q, want %q", got, "5")
	}
	if got := coerceMetadataValue(float64(5.25)); got != "5.25" {
		t.Errorf("coerce float decimal: got %q, want %q", got, "5.25")
	}
	if got := coerceMetadataValue(nil); got != "null" {
		t.Errorf("coerce nil: got %q, want %q", got, "null")
	}
	if got := coerceWireMetadata(nil); got != nil {
		t.Errorf("coerce nil map: got %v, want nil", got)
	}
	if got := encodeWireMetadata(nil); got != nil {
		t.Errorf("encode nil map: got %v, want nil", got)
	}

	encoded := encodeWireMetadata(map[string]string{"k": "v"})
	if encoded["k"] != "v" {
		t.Errorf("encode map: got %v, want map[k:v]", encoded)
	}
}
