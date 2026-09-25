package extproc

import (
	"encoding/json"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const azureIngressTestConfig = `
version: v0.3
providers:
  models:
    - name: worker
      provider_model_id: worker-prod
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: http://127.0.0.1:18000/v1
          api_key: provider-secret
    - name: other
      provider_model_id: other-prod
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: http://127.0.0.1:18001/v1
          api_key: provider-secret
routing: {}
`

func TestValidateAzureDeploymentPaths(t *testing.T) {
	router := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore())}
	tests := []struct {
		method string
		path   string
		status typev3.StatusCode
	}{
		{method: "POST", path: "/openai/deployments/copilot-auto/chat/completions?api-version=2024-10-21"},
		{method: "POST", path: "/openai/deployments/copilot-auto/chat/completions?api-version=2025-04-01-preview&trace=1"},
		{method: "POST", path: "/openai/deployments/vllm-sr/auto/chat/completions?api-version=2024-10-21"},
		{method: "POST", path: "/openai/deployments/copilot-auto/chat/completions/"},
		{method: "GET", path: "/openai/deployments/copilot-auto/chat/completions?api-version=2024-10-21", status: typev3.StatusCode_MethodNotAllowed},
		{method: "POST", path: "/openai/deployments/copilot-auto/embeddings?api-version=2024-10-21", status: typev3.StatusCode_NotFound},
		{method: "POST", path: "/openai/deployments//chat/completions?api-version=2024-10-21", status: typev3.StatusCode_NotFound},
		{method: "GET", path: "/openai/deployments?api-version=2024-10-21", status: typev3.StatusCode_NotFound},
		{method: "POST", path: "/openai/v1/chat/completions"},
		{method: "POST", path: "/openai/v1/responses"},
		{method: "POST", path: "/openai/responses?api-version=2025-04-01-preview"},
		{method: "GET", path: "/openai/v1/chat/completions", status: typev3.StatusCode_MethodNotAllowed},
		{method: "GET", path: "/openai/responses?api-version=2025-04-01-preview", status: typev3.StatusCode_MethodNotAllowed},
		{method: "POST", path: "/openai/v1/embeddings", status: typev3.StatusCode_NotFound},
		{method: "GET", path: "/openai/v1/models", status: typev3.StatusCode_NotFound},
		{method: "GET", path: "/openai/v1/responses/resp_123", status: typev3.StatusCode_NotFound},
		{method: "POST", path: "/openai/chat/completions?api-version=2024-10-21", status: typev3.StatusCode_NotFound},
		{method: "POST", path: "/openai", status: typev3.StatusCode_NotFound},
	}
	for _, test := range tests {
		t.Run(test.method+" "+test.path, func(t *testing.T) {
			response := router.validateRequestHeaders(test.method, test.path)
			if test.status == 0 {
				assert.Nil(t, response)
				return
			}
			require.NotNil(t, response.GetImmediateResponse(), "path was not rejected")
			assert.Equal(t, test.status, response.GetImmediateResponse().GetStatus().GetCode())
		})
	}
}

func TestAzureDeploymentChatRoutesTheDeploymentModel(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "body without model", body: `{"messages":[{"role":"user","content":"hello"}]}`},
		{name: "body naming another model", body: `{"model":"other","messages":[{"role":"user","content":"hello"}]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream := NewMockStream([]*ext_proc.ProcessingRequest{
				azureIngressHeaders("/openai/deployments/worker/chat/completions?api-version=2024-10-21"),
				{Request: &ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{
					Body: []byte(test.body), EndOfStream: true,
				}}},
			})
			require.NoError(t, newAzureIngressTestRouter(t).Process(stream))
			require.Len(t, stream.Responses, 2)

			removed := stream.Responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders()
			assert.Contains(t, removed, "api-key", "the client key must not reach the provider")

			dispatch := stream.Responses[1].GetRequestBody().GetResponse()
			require.NotNil(t, dispatch, "request was not dispatched: %v", stream.Responses[1].GetImmediateResponse())
			emitted := headerValuesByName(dispatch.GetHeaderMutation().GetSetHeaders())
			assert.Equal(t, "worker", emitted[headers.SelectedModel])
			assert.Equal(t, "/v1/chat/completions", emitted[":path"])
			var wire struct {
				Model string `json:"model"`
			}
			require.NoError(t, json.Unmarshal(dispatch.GetBodyMutation().GetBody(), &wire))
			assert.Equal(t, "worker-prod", wire.Model)
		})
	}
}

func TestAzureBodyModelPathsDispatch(t *testing.T) {
	tests := []struct {
		name string
		path string
		body string
	}{
		{name: "v1 chat completions", path: "/openai/v1/chat/completions", body: `{"model":"worker","messages":[{"role":"user","content":"hello"}]}`},
		{name: "responses with api-version", path: "/openai/responses?api-version=2025-04-01-preview", body: `{"model":"worker","input":"hello"}`},
		{name: "v1 responses", path: "/openai/v1/responses", body: `{"model":"worker","input":"hello"}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			router := newAzureIngressTestRouter(t)
			router.ResponseAPIFilter = NewResponseAPIFilter(NewMockResponseStore())
			stream := NewMockStream([]*ext_proc.ProcessingRequest{
				azureIngressHeaders(test.path),
				{Request: &ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{
					Body: []byte(test.body), EndOfStream: true,
				}}},
			})
			require.NoError(t, router.Process(stream))
			require.Len(t, stream.Responses, 2)

			removed := stream.Responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders()
			assert.Contains(t, removed, "api-key", "the client key must not reach the provider")

			dispatch := stream.Responses[1].GetRequestBody().GetResponse()
			require.NotNil(t, dispatch, "request was not dispatched: %v", stream.Responses[1].GetImmediateResponse())
			emitted := headerValuesByName(dispatch.GetHeaderMutation().GetSetHeaders())
			assert.Equal(t, "worker", emitted[headers.SelectedModel])
			assert.Equal(t, "/v1/chat/completions", emitted[":path"])
			var wire struct {
				Model    string `json:"model"`
				Messages []struct {
					Role    string `json:"role"`
					Content string `json:"content"`
				} `json:"messages"`
			}
			require.NoError(t, json.Unmarshal(dispatch.GetBodyMutation().GetBody(), &wire))
			assert.Equal(t, "worker-prod", wire.Model)
			require.Len(t, wire.Messages, 1)
			assert.Equal(t, "user", wire.Messages[0].Role)
			assert.Equal(t, "hello", wire.Messages[0].Content)
		})
	}
}

func TestAzureResponsesPathsFollowTheResponsesAPISetting(t *testing.T) {
	router := &OpenAIRouter{}
	for _, path := range []string{"/v1/responses", "/openai/v1/responses", "/openai/responses?api-version=2025-04-01-preview"} {
		t.Run(path, func(t *testing.T) {
			response := router.validateRequestHeaders("POST", path)
			require.NotNil(t, response.GetImmediateResponse(), "path was not rejected")
			assert.Equal(t, typev3.StatusCode_NotFound, response.GetImmediateResponse().GetStatus().GetCode())
		})
	}
}

func TestAzureRejectsUnsupportedOperation(t *testing.T) {
	for _, path := range []string{
		"/openai/deployments/worker/embeddings?api-version=2024-10-21",
		"/openai/v1/embeddings",
		"/openai/v1/responses/resp_123",
		"/openai/images/generations?api-version=2024-10-21",
	} {
		t.Run(path, func(t *testing.T) {
			stream := NewMockStream([]*ext_proc.ProcessingRequest{azureIngressHeaders(path)})
			require.NoError(t, newAzureIngressTestRouter(t).Process(stream))
			require.Len(t, stream.Responses, 1)
			immediate := stream.Responses[0].GetImmediateResponse()
			require.NotNil(t, immediate, "unsupported Azure operation was processed as Chat Completions")
			assert.Equal(t, typev3.StatusCode_NotFound, immediate.GetStatus().GetCode())
		})
	}
}

func newAzureIngressTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(azureIngressTestConfig))
	require.NoError(t, err)
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	require.NoError(t, err)
	return &OpenAIRouter{
		Config:             cfg,
		Classifier:         classifier,
		Cache:              cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false}),
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
	}
}

func azureIngressHeaders(path string) *ext_proc.ProcessingRequest {
	return &ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: path},
			{Key: "content-type", Value: "application/json"},
			{Key: "api-key", Value: "client-key"},
		}}},
	}}
}
