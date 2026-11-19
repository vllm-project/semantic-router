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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

// chatStreamUsageConfig serves one model from a backend of each format.
const chatStreamUsageConfig = `
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
    - name: messages-model
      api_format: anthropic
      backend_refs:
        - name: messages-backend
          endpoint: 127.0.0.1:18102
          protocol: http
          provider: anthropic
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
`

// vLLM's chunks for a stream the Router asks for usage: none carries a usage
// member until the usage chunk.
var vllmChatStreamChunks = []string{
	"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"chat-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n",
	"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"chat-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Paris\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n",
	"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"chat-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"logprobs\":null,\"finish_reason\":\"stop\",\"stop_reason\":null}]}\n\n",
	"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"chat-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"total_tokens\":6,\"completion_tokens\":1}}\n\n",
	"data: [DONE]\n\n",
}

func streamedChatClientBody(t *testing.T, model string, includeUsage bool, upstream *parity.Upstream) []byte {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(chatStreamUsageConfig))
	require.NoError(t, err)
	options := ""
	if includeUsage {
		options = `"stream_options":{"include_usage":true},`
	}
	record := parity.NewRecorder(newParityRouter(t, cfg), routing.DefaultOptions).Run(context.Background(), parity.Case{
		Name: t.Name(),
		Request: parity.CaseRequest{
			Method: "POST", Path: "/v1/chat/completions",
			Headers: [][]string{{"content-type", "application/json"}, {"accept", "text/event-stream"}},
			Body:    `{"model":"` + model + `","stream":true,` + options + `"messages":[{"role":"user","content":"What is the capital of France?"}]}`,
		},
		Upstream: upstream,
	})
	require.Empty(t, record.Error)
	require.NotNil(t, record.Upstream)
	require.Equal(t, http.StatusOK, record.Response.Status, "%s", record.Response.Body)
	body := append([]byte(nil), record.Response.Body...)
	for _, chunk := range record.Response.Chunks {
		body = append(body, chunk...)
	}
	return body
}

// streamUsages names each chunk's usage member: "absent", "null" or "totals".
func streamUsages(t *testing.T, stream []byte) []string {
	t.Helper()
	var usages []string
	for _, frame := range strings.Split(string(stream), "\n\n") {
		data, found := strings.CutPrefix(frame, "data: ")
		if !found || data == "[DONE]" {
			continue
		}
		var chunk map[string]json.RawMessage
		require.NoError(t, json.Unmarshal([]byte(data), &chunk), "%s", data)
		switch usage, present := chunk["usage"]; {
		case !present:
			usages = append(usages, "absent")
		case string(usage) == "null":
			usages = append(usages, "null")
		default:
			require.JSONEq(t, `[]`, string(chunk["choices"]), "the usage chunk: %s", data)
			usages = append(usages, "totals")
		}
	}
	return usages
}

func requireNullUsageUntilTheUsageChunk(t *testing.T, stream []byte) {
	t.Helper()
	usages := streamUsages(t, stream)
	require.Greater(t, len(usages), 1, "%s", stream)
	require.Equal(t, "totals", usages[len(usages)-1], "%s", stream)
	for index, usage := range usages[:len(usages)-1] {
		require.Equal(t, "null", usage, "chunk %d of %s", index, stream)
	}
}

// A streamed Chat client that sets include_usage gets "usage": null on every
// chunk before the usage chunk, as OpenAI sends it, from a backend of its own
// format and from the others; without include_usage no chunk has usage.
func TestChatStreamCarriesNullUsageUntilTheUsageChunk(t *testing.T) {
	sse := func(chunks ...string) *parity.Upstream {
		return &parity.Upstream{Status: http.StatusOK, Headers: [][]string{{"content-type", "text/event-stream"}}, Chunks: chunks}
	}
	for _, test := range []struct {
		name, model string
		upstream    *parity.Upstream
	}{
		{"chat backend", "chat-model", sse(vllmChatStreamChunks...)},
		{"responses backend", "responses-model", sse(string(extProcStreamFixture(llmprotocol.OpenAIResponsesV1)))},
		{"messages backend", "messages-model", sse(string(extProcStreamFixture(llmprotocol.AnthropicMessagesV1)))},
	} {
		t.Run(test.name, func(t *testing.T) {
			body := streamedChatClientBody(t, test.model, true, test.upstream)
			requireNullUsageUntilTheUsageChunk(t, body)
			require.True(t, bytes.HasSuffix(body, []byte("data: [DONE]\n\n")), "%s", body)

			for index, usage := range streamUsages(t, streamedChatClientBody(t, test.model, false, test.upstream)) {
				require.Equal(t, "absent", usage, "chunk %d", index)
			}
		})
	}
}

// From a Chat backend, the client's chunks are the backend's bytes with only
// the null usage added. The record zeroes clock fields, so the expected
// chunks are compared as it keeps them.
func TestChatBackendChunksGainOnlyTheNullUsage(t *testing.T) {
	body := streamedChatClientBody(t, "chat-model", true, &parity.Upstream{
		Status: http.StatusOK, Headers: [][]string{{"content-type", "text/event-stream"}}, Chunks: vllmChatStreamChunks,
	})
	content := strings.TrimSuffix(vllmChatStreamChunks[1], "}\n\n") + `,"usage":null}` + "\n\n"
	require.Contains(t, string(body), string(parity.NormalizeBody([]byte(content))))
	require.Contains(t, string(body), string(parity.NormalizeBody([]byte(vllmChatStreamChunks[3]))), "the usage chunk changed")
}
