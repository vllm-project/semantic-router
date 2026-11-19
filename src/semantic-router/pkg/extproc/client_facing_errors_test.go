package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

// clientErrorsConfig serves chat-model and other-model from vLLM Chat
// backends, and Responses clients too.
const clientErrorsConfig = `
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
    - name: other-model
      api_format: chat_completions
      backend_refs:
        - name: other-backend
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

// vLLM's 400 bodies: the flat error of releases before 0.12 for an output
// limit beyond the window, and the nested one since for max_tokens 0 and
// for an output limit beyond the window.
const (
	vllmLegacyOutputLimitBody = `{"object":"error","message":"'max_tokens' or 'max_completion_tokens' is too large: 5000000. This model's maximum context length is 262144 tokens and your request has 12 input tokens (5000000 > 262144 - 12). None","type":"BadRequestError","param":null,"code":400}`
	vllmOutputLimitBody       = `{"error":{"message":"max_completion_tokens=5000000 cannot be greater than max_model_len=262144. Please request fewer output tokens. (parameter=max_completion_tokens, value=5000000)","type":"BadRequestError","param":"max_completion_tokens","code":400}}`
	vllmZeroOutputBody        = `{"error":{"message":"max_tokens must be at least 1, got 0. (parameter=max_tokens, value=0)","type":"BadRequestError","param":"max_tokens","code":400}}`
)

type clientErrorBody struct {
	Type  string `json:"type"`
	Error struct {
		Type    string  `json:"type"`
		Code    *string `json:"code"`
		Message string  `json:"message"`
		Param   *string `json:"param"`
	} `json:"error"`
}

func runClientErrorCase(t *testing.T, ctx context.Context, path, body string, upstream *parity.Upstream) *parity.Record {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(clientErrorsConfig))
	require.NoError(t, err)
	headers := [][]string{{"content-type", "application/json"}}
	if path == "/v1/messages" {
		headers = append(headers, []string{"anthropic-version", "2023-06-01"})
	}
	record := parity.NewRecorder(newParityRouter(t, cfg), routing.DefaultOptions).Run(ctx, parity.Case{
		Name:     t.Name(),
		Request:  parity.CaseRequest{Method: "POST", Path: path, Headers: headers, Body: body},
		Upstream: upstream,
	})
	require.Empty(t, record.Error)
	require.NotNil(t, record.Response)
	return record
}

func decodeClientErrorBody(t *testing.T, record *parity.Record) clientErrorBody {
	t.Helper()
	var decoded clientErrorBody
	require.NoError(t, json.Unmarshal(record.Response.Body, &decoded), "%s", record.Response.Body)
	return decoded
}

func badRequestFrom(body string) *parity.Upstream {
	return &parity.Upstream{Status: http.StatusBadRequest, Headers: [][]string{{"content-type", "application/json"}}, Body: body}
}

// A routed Chat client's max_tokens reaches a vLLM Chat backend as the
// max_completion_tokens the Router sends it. The backend's 400 comes back as
// context_length_exceeded or invalid_request, naming max_tokens, with the
// window and the counts in plain words, from either vLLM error shape.
func TestBackendRefusalsNameTheChatClientsField(t *testing.T) {
	for _, test := range []struct {
		name, maxTokens, upstream, code, message string
	}{
		{
			"legacy vLLM, output beyond the window", "5000000", vllmLegacyOutputLimitBody, "context_length_exceeded",
			"max_tokens is too large: 5000000. This model's maximum context length is 262144 tokens, and your request has 12 input tokens. Please request fewer output tokens.",
		},
		{
			"vLLM, output beyond the window", "5000000", vllmOutputLimitBody, "context_length_exceeded",
			"max_tokens is too large: 5000000. This model's maximum context length is 262144 tokens. Please request fewer output tokens.",
		},
		{"vLLM, max_tokens 0", "0", vllmZeroOutputBody, "invalid_request", "max_tokens must be at least 1, got 0."},
	} {
		t.Run(test.name, func(t *testing.T) {
			record := runClientErrorCase(t, context.Background(), "/v1/chat/completions",
				`{"model":"vllm-sr/auto","max_tokens":`+test.maxTokens+`,"messages":[{"role":"user","content":"What is the capital of Spain?"}]}`,
				badRequestFrom(test.upstream))
			require.NotNil(t, record.Upstream)
			require.Contains(t, string(record.Upstream.Body), `"max_completion_tokens":`+test.maxTokens, "the Chat backend gets the Router's field")
			require.Equal(t, http.StatusBadRequest, record.Response.Status)
			decoded := decodeClientErrorBody(t, record)
			require.Equal(t, "invalid_request_error", decoded.Error.Type)
			require.Equal(t, test.code, *decoded.Error.Code)
			require.Equal(t, "max_tokens", *decoded.Error.Param)
			require.Equal(t, test.message, decoded.Error.Message)
			require.NotContains(t, string(record.Response.Body), "max_model_len")
			require.NotContains(t, string(record.Response.Body), "max_completion_tokens")
		})
	}
}

// Responses and Messages clients get the same refusal in their envelopes,
// under their own name for the output limit.
func TestBackendRefusalsNameTheResponsesAndMessagesClientsField(t *testing.T) {
	record := runClientErrorCase(t, context.Background(), "/v1/responses",
		`{"model":"chat-model","store":false,"max_output_tokens":5000000,"input":"What is the capital of Spain?"}`,
		badRequestFrom(vllmOutputLimitBody))
	require.Equal(t, http.StatusBadRequest, record.Response.Status)
	decoded := decodeClientErrorBody(t, record)
	require.Equal(t, "context_length_exceeded", *decoded.Error.Code)
	require.Equal(t, "max_output_tokens", *decoded.Error.Param)
	require.Equal(t, "max_output_tokens is too large: 5000000. This model's maximum context length is 262144 tokens. Please request fewer output tokens.", decoded.Error.Message)

	record = runClientErrorCase(t, context.Background(), "/v1/messages",
		`{"model":"chat-model","max_tokens":5000000,"messages":[{"role":"user","content":"What is the capital of Spain?"}]}`,
		badRequestFrom(vllmLegacyOutputLimitBody))
	require.Equal(t, http.StatusBadRequest, record.Response.Status)
	decoded = decodeClientErrorBody(t, record)
	require.Equal(t, "error", decoded.Type)
	require.Equal(t, "invalid_request_error", decoded.Error.Type)
	require.Equal(t, "max_tokens is too large: 5000000. This model's maximum context length is 262144 tokens, and your request has 12 input tokens. Please request fewer output tokens.", decoded.Error.Message)
}

// Valid JSON with a decimal max_tokens or a member the API does not define
// is refused with the member named, as OpenAI and Anthropic refuse it, and
// never reaches a backend.
func TestRequestFieldRefusalsNameTheField(t *testing.T) {
	for _, test := range []struct {
		name, path, body, code, param, message string
	}{
		{
			"chat decimal", "/v1/chat/completions", `{"model":"chat-model","max_tokens":5.5,"messages":[{"role":"user","content":"hi"}]}`,
			"invalid_type", "max_tokens", "Invalid type for 'max_tokens': expected an integer, but got a decimal number instead.",
		},
		{
			"chat unknown member", "/v1/chat/completions", `{"model":"chat-model","foo":1,"messages":[{"role":"user","content":"hi"}]}`,
			"unknown_parameter", "foo", "Unrecognized request argument supplied: foo",
		},
		{
			"responses unknown member", "/v1/responses", `{"model":"chat-model","store":false,"foo":1,"input":"hi"}`,
			"unknown_parameter", "foo", "Unrecognized request argument supplied: foo",
		},
		{
			"messages decimal", "/v1/messages", `{"model":"chat-model","max_tokens":5.5,"messages":[{"role":"user","content":"hi"}]}`,
			"", "", "Invalid type for 'max_tokens': expected an integer, but got a decimal number instead.",
		},
		{
			"messages unknown member", "/v1/messages", `{"model":"chat-model","max_tokens":8,"foo":1,"messages":[{"role":"user","content":"hi"}]}`,
			"", "", "Unrecognized request argument supplied: foo",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			record := runClientErrorCase(t, context.Background(), test.path, test.body, nil)
			require.Nil(t, record.Upstream)
			require.Equal(t, http.StatusBadRequest, record.Response.Status)
			decoded := decodeClientErrorBody(t, record)
			require.Equal(t, "invalid_request_error", decoded.Error.Type)
			require.Equal(t, test.message, decoded.Error.Message)
			if test.code == "" {
				require.Equal(t, "error", decoded.Type, "an Anthropic envelope")
				require.Nil(t, decoded.Error.Code)
				return
			}
			require.Equal(t, test.code, *decoded.Error.Code)
			require.Equal(t, test.param, *decoded.Error.Param)
		})
	}
}

// On a listener that publishes some models, a model the Router serves
// elsewhere is withheld (403 model_not_allowed), and one it serves nowhere is
// unknown (404 model_not_found), as OpenAI SDKs expect; a Messages client
// gets Anthropic's not_found_error.
func TestUnknownAndUnpublishedModelsOnAListener(t *testing.T) {
	listener := routing.WithListenerModels(context.Background(), []string{"vllm-sr/auto", "chat-model"})
	request := func(model string) string {
		return `{"model":"` + model + `","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`
	}
	for _, test := range []struct {
		name, path, model string
		status            int
		errorType, code   string
		message           string
	}{
		{"unpublished", "/v1/chat/completions", "other-model", http.StatusForbidden, "permission_error", "model_not_allowed", `The model "other-model" is not available on this endpoint.`},
		{"unknown", "/v1/chat/completions", "no-such-model", http.StatusNotFound, "invalid_request_error", "model_not_found", "The model `no-such-model` does not exist or you do not have access to it."},
		{"unknown to a Messages client", "/v1/messages", "no-such-model", http.StatusNotFound, "not_found_error", "", "The model `no-such-model` does not exist or you do not have access to it."},
	} {
		t.Run(test.name, func(t *testing.T) {
			record := runClientErrorCase(t, listener, test.path, request(test.model), nil)
			require.Nil(t, record.Upstream)
			require.Equal(t, test.status, record.Response.Status)
			decoded := decodeClientErrorBody(t, record)
			require.Equal(t, test.errorType, decoded.Error.Type)
			require.Equal(t, test.message, decoded.Error.Message)
			if test.code != "" {
				require.Equal(t, test.code, *decoded.Error.Code)
			}
		})
	}
	record := runClientErrorCase(t, context.Background(), "/v1/chat/completions", request("no-such-model"), nil)
	require.Equal(t, http.StatusNotFound, record.Response.Status, "the Router's own unknown model answer")
	require.Equal(t, "model_not_found", *decodeClientErrorBody(t, record).Error.Code)
}
