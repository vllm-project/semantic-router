package protocolcodec

import (
	"bytes"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Chat or Responses request may send a value of a field the Router cannot
// carry that asks for what the backend does anyway: the request decodes, the
// value is reported as dropped, and no backend body carries it, including a
// same-format one that would otherwise replay the client's bytes.
func TestNoOpRequestValuesAreDropped(t *testing.T) {
	const chatMessages = `"messages":[{"role":"user","content":"hi"}]`
	for _, test := range []struct {
		name    string
		format  llmprotocol.WireFormat
		body    string
		dropped []string
	}{
		{"chat auto tier", llmprotocol.OpenAIChatV1, `{"model":"m",` + chatMessages + `,"service_tier":"auto"}`, []string{"service_tier"}},
		{"chat default tier", llmprotocol.OpenAIChatV1, `{"model":"m",` + chatMessages + `,"service_tier":"default"}`, []string{"service_tier"}},
		{"chat text modalities", llmprotocol.OpenAIChatV1, `{"model":"m",` + chatMessages + `,"modalities":["text"]}`, []string{"modalities"}},
		{"chat safety identifier", llmprotocol.OpenAIChatV1, `{"model":"m",` + chatMessages + `,"safety_identifier":"user-1234"}`, []string{"safety_identifier"}},
		{
			"chat all three", llmprotocol.OpenAIChatV1,
			`{"model":"m",` + chatMessages + `,"service_tier":"auto","modalities":["text"],"safety_identifier":"user-1234"}`,
			[]string{"modalities", "safety_identifier", "service_tier"},
		},
		{"responses default tier", llmprotocol.OpenAIResponsesV1, `{"model":"m","input":"hi","service_tier":"default"}`, []string{"service_tier"}},
		{
			"responses tier and identifier", llmprotocol.OpenAIResponsesV1,
			`{"model":"m","input":"hi","service_tier":"auto","safety_identifier":"user-1234"}`,
			[]string{"safety_identifier", "service_tier"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			engine := NewBuiltinEngine()
			request, envelope, diagnostics, err := engine.DecodeRequest(test.format, []byte(test.body))
			if err != nil {
				t.Fatal(err)
			}
			var dropped []string
			for _, diagnostic := range diagnostics {
				if diagnostic.Action != llmprotocol.DiagnosticDropped || diagnostic.Source != test.format || diagnostic.Reason == "" {
					t.Fatalf("diagnostic = %+v", diagnostic)
				}
				dropped = append(dropped, diagnostic.Field)
			}
			if !reflect.DeepEqual(dropped, test.dropped) {
				t.Fatalf("dropped %v, want %v", dropped, test.dropped)
			}
			encoded, err := engine.EncodeRequest(test.format, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			for _, field := range test.dropped {
				if bytes.Contains(encoded.Body, []byte(`"`+field+`"`)) {
					t.Fatalf("the backend body carries %s: %s", field, encoded.Body)
				}
			}
		})
	}
}

// Any other value of those fields is still refused.
func TestOtherValuesOfNoOpFieldsAreRefused(t *testing.T) {
	const chatMessages = `"messages":[{"role":"user","content":"hi"}]`
	for _, test := range []struct {
		name, body, code string
		format           llmprotocol.WireFormat
	}{
		{"chat flex tier", `{"model":"m",` + chatMessages + `,"service_tier":"flex"}`, "unsupported_service_tier", llmprotocol.OpenAIChatV1},
		{"chat priority tier", `{"model":"m",` + chatMessages + `,"service_tier":"priority"}`, "unsupported_service_tier", llmprotocol.OpenAIChatV1},
		{"chat audio output", `{"model":"m",` + chatMessages + `,"modalities":["text","audio"]}`, "unsupported_modalities", llmprotocol.OpenAIChatV1},
		{"chat no modalities", `{"model":"m",` + chatMessages + `,"modalities":[]}`, "unsupported_modalities", llmprotocol.OpenAIChatV1},
		{"chat numeric identifier", `{"model":"m",` + chatMessages + `,"safety_identifier":1234}`, "unsupported_safety_identifier", llmprotocol.OpenAIChatV1},
		{"chat logprobs", `{"model":"m",` + chatMessages + `,"logprobs":true}`, "unsupported_logprobs", llmprotocol.OpenAIChatV1},
		{"responses scale tier", `{"model":"m","input":"hi","service_tier":"scale"}`, "unsupported_service_tier", llmprotocol.OpenAIResponsesV1},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, _, _, err := NewBuiltinEngine().DecodeRequest(test.format, []byte(test.body))
			var protocolError *llmprotocol.ProtocolError
			if !errors.As(err, &protocolError) || protocolError.Code != test.code {
				t.Fatalf("err = %v, want %s", err, test.code)
			}
		})
	}
}
