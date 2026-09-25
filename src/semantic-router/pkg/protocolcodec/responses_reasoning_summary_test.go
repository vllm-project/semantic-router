package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Copilot CLI 1.0.88 asks for a reasoning summary on every Responses turn unless
// it runs with --reasoning-effort none.
const copilotReasoningSummaryRequest = `{"model":"worker","input":"hello","stream":true,"reasoning":{"effort":"medium","summary":"auto"}}`

func decodeRoutedReasoningSummaryRequest(t *testing.T, engine *Engine, body string) (llmprotocol.Request, llmprotocol.Envelope) {
	t.Helper()
	request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
	if err != nil {
		t.Fatalf("Responses request with a reasoning summary was rejected: %v", err)
	}
	request.Model = "routed-model"
	request.Generation++
	return request, envelope
}

func TestResponsesReasoningSummaryReachesResponsesBackends(t *testing.T) {
	engine := NewBuiltinEngine()
	request, envelope := decodeRoutedReasoningSummaryRequest(t, engine, copilotReasoningSummaryRequest)
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var dispatch struct {
		Reasoning map[string]string `json:"reasoning"`
	}
	if err := json.Unmarshal(encoded.Body, &dispatch); err != nil {
		t.Fatal(err)
	}
	if dispatch.Reasoning["summary"] != "auto" || dispatch.Reasoning["effort"] != "medium" {
		t.Fatalf("Responses dispatch reasoning = %v, want effort medium and summary auto", dispatch.Reasoning)
	}
	if len(encoded.Diagnostics) != 0 {
		t.Fatalf("Responses dispatch reported %+v", encoded.Diagnostics)
	}
}

func TestResponsesReasoningSummaryIsDroppedForChatAndMessages(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1} {
		t.Run(string(format), func(t *testing.T) {
			request, envelope := decodeRoutedReasoningSummaryRequest(t, engine, copilotReasoningSummaryRequest)
			encoded, err := engine.EncodeRequest(format, request, envelope)
			if err != nil {
				t.Fatalf("dispatch to %s failed: %v", format, err)
			}
			if bytes.Contains(encoded.Body, []byte(`"summary"`)) {
				t.Fatalf("dispatch to %s carried the summary: %s", format, encoded.Body)
			}
			for _, diagnostic := range encoded.Diagnostics {
				if diagnostic.Field == "reasoning.summary" && diagnostic.Action == llmprotocol.DiagnosticDropped {
					return
				}
			}
			t.Fatalf("dispatch to %s did not report the dropped summary: %+v", format, encoded.Diagnostics)
		})
	}
}

func TestResponsesReasoningSummaryValues(t *testing.T) {
	tests := []struct {
		summary string
		want    string
		code    string
	}{
		{summary: `"auto"`, want: "auto"},
		{summary: `"concise"`, want: "concise"},
		{summary: `"detailed"`, want: "detailed"},
		{summary: `null`},
		{summary: `"verbose"`, code: "invalid_reasoning_summary"},
		{summary: `1`, code: "invalid_reasoning_summary"},
	}
	engine := NewBuiltinEngine()
	for _, test := range tests {
		t.Run(test.summary, func(t *testing.T) {
			body := `{"model":"m","input":"hello","reasoning":{"effort":"low","summary":` + test.summary + `}}`
			if test.code != "" {
				_, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, []byte(body))
				var protocolError *llmprotocol.ProtocolError
				if !errors.As(err, &protocolError) || protocolError.Code != test.code {
					t.Fatalf("summary %s returned %v, want %s", test.summary, err, test.code)
				}
				return
			}
			request, envelope := decodeRoutedReasoningSummaryRequest(t, engine, body)
			encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			var dispatch struct {
				Reasoning map[string]string `json:"reasoning"`
			}
			if err := json.Unmarshal(encoded.Body, &dispatch); err != nil {
				t.Fatal(err)
			}
			if dispatch.Reasoning["summary"] != test.want {
				t.Fatalf("summary %s reached Responses as %q, want %q", test.summary, dispatch.Reasoning["summary"], test.want)
			}
		})
	}
}
