package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

// noOpFieldsConfig serves chat-model over Chat Completions and
// responses-model over Responses, from vLLM backends.
const noOpFieldsConfig = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
providers:
  models:
    - name: chat-model
      api_format: chat_completions
      backend_refs:
        - name: chat-backend
          endpoint: 127.0.0.1:18100
          protocol: http
          provider: vllm
    - name: responses-model
      api_format: responses
      backend_refs:
        - name: responses-backend
          endpoint: 127.0.0.1:18101
          protocol: http
          provider: vllm
recipes:
  - name: default
    decisions:
      - name: default_route
        priority: 10
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: chat-model
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
global:
  services:
    response_api:
      enabled: true
      store_backend: memory
`

func runNoOpFieldsCase(t *testing.T, path, body string, backend llmprotocol.WireFormat) *parity.Record {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(noOpFieldsConfig))
	require.NoError(t, err)
	record := parity.NewRecorder(newParityRouter(t, cfg), routing.DefaultOptions).Run(context.Background(), parity.Case{
		Name:    t.Name(),
		Request: parity.CaseRequest{Method: "POST", Path: path, Headers: [][]string{{"content-type", "application/json"}}, Body: body},
		Upstream: &parity.Upstream{
			Status: http.StatusOK, Headers: [][]string{{"content-type", "application/json"}}, Body: string(extProcResponseFixture(backend)),
		},
	})
	require.Empty(t, record.Error)
	require.NotNil(t, record.Response)
	return record
}

func droppedFieldWarning(reason, field string) string {
	return `protocol;action=dropped;reason="` + reason + `";field="` + field + `"`
}

// A service_tier of auto or default, modalities of only text and a
// safety_identifier ask a backend for nothing it would not do anyway. A Chat
// or Responses request that carries them is served, its backend receives none
// of them, even from a backend of the client's own format, which would
// otherwise receive the client's bytes, and the answer's x-vsr-warnings
// reports each as dropped.
func TestNoOpRequestFieldsAreDroppedWithAWarning(t *testing.T) {
	tier := droppedFieldWarning("the auto and default service tiers ask for the backend's standard processing", "service_tier")
	modalities := droppedFieldWarning("text is the only output a backend returns without modalities", "modalities")
	identifier := droppedFieldWarning("an end-user identifier for the provider's abuse monitoring is not model input", "safety_identifier")
	for _, test := range []struct {
		name, path, body string
		backend          llmprotocol.WireFormat
		warnings         []string
	}{
		{
			"chat to a Chat backend", "/v1/chat/completions",
			`{"model":"chat-model","service_tier":"auto","modalities":["text"],"safety_identifier":"user-1234","messages":[{"role":"user","content":"hi"}]}`,
			llmprotocol.OpenAIChatV1,
			[]string{modalities, identifier, tier},
		},
		{
			"responses to a Responses backend", "/v1/responses",
			`{"model":"responses-model","store":false,"service_tier":"default","safety_identifier":"user-1234","input":"hi"}`,
			llmprotocol.OpenAIResponsesV1,
			[]string{identifier, tier},
		},
		{
			"responses to a Chat backend", "/v1/responses",
			`{"model":"chat-model","store":false,"service_tier":"auto","input":"hi"}`,
			llmprotocol.OpenAIChatV1,
			[]string{tier},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			record := runNoOpFieldsCase(t, test.path, test.body, test.backend)
			require.Equal(t, http.StatusOK, record.Response.Status, "%s", record.Response.Body)
			require.NotNil(t, record.Upstream)
			for _, field := range []string{"service_tier", "modalities", "safety_identifier"} {
				require.False(t, bytes.Contains(record.Upstream.Body, []byte(`"`+field+`"`)), "the backend received %s: %s", field, record.Upstream.Body)
			}
			warnings := record.Response.Header.Get(headers.VSRWarnings)
			for _, warning := range test.warnings {
				require.Contains(t, warnings, warning)
			}
			for _, field := range []string{"service_tier", "modalities", "safety_identifier"} {
				reported := 0
				if strings.Contains(strings.Join(test.warnings, ","), `field="`+field+`"`) {
					reported = 1
				}
				require.Equal(t, reported, strings.Count(warnings, `field="`+field+`"`), "x-vsr-warnings = %q", warnings)
			}
		})
	}
}

// Any other value of those fields still fails closed before any backend.
func TestOtherValuesOfNoOpFieldsAreStillRefused(t *testing.T) {
	for _, test := range []struct {
		name, path, body, code string
	}{
		{"chat flex tier", "/v1/chat/completions", `{"model":"chat-model","service_tier":"flex","messages":[{"role":"user","content":"hi"}]}`, "unsupported_service_tier"},
		{"chat audio output", "/v1/chat/completions", `{"model":"chat-model","modalities":["text","audio"],"messages":[{"role":"user","content":"hi"}]}`, "unsupported_modalities"},
		{"responses priority tier", "/v1/responses", `{"model":"responses-model","store":false,"service_tier":"priority","input":"hi"}`, "unsupported_service_tier"},
	} {
		t.Run(test.name, func(t *testing.T) {
			record := runNoOpFieldsCase(t, test.path, test.body, llmprotocol.OpenAIChatV1)
			require.Nil(t, record.Upstream)
			require.Equal(t, http.StatusBadRequest, record.Response.Status)
			var body struct {
				Error struct {
					Code string `json:"code"`
				} `json:"error"`
			}
			require.NoError(t, json.Unmarshal(record.Response.Body, &body), "%s", record.Response.Body)
			require.Equal(t, test.code, body.Error.Code)
		})
	}
}
