package extproc

import (
	"encoding/json"
	"fmt"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Compression always drops the middle sentence: it alone exceeds the 16-token
// budget and holds the code and the only "zanzibar". The decision needs that
// sentence absent, so a coding-model route proves the classifier saw
// compressed text.
const (
	compressionAgentPrompt = "Refactor the config loader in this repository. Here is the current code:\n" +
		"```go\nfunc ParseConfig(path string) (*Config, error) {\n\tdata, err := readFile(path)\n" +
		"\tif err == nil {\n\t\treturn decodeConfig(data)\n\t}\n\treturn nil, wrapReadError(path, err)\n}\n```\n" +
		"Keep the exported ParseConfig signature, update every caller, and leave the zanzibar flag default unchanged. " +
		"Run the full test suite with the race detector before you report back."
	compressionAgentInstructions    = "You are a coding agent. Edit files only through the apply_patch tool."
	compressionAgentToolDescription = "Apply a unified diff to files in the workspace."
	compressionAgentToolSchema      = `{"type":"object","properties":{"patch":{"type":"string","description":"Unified diff to apply."}},"required":["patch"],"additionalProperties":false}`
	compressionRoutedUpstreamModel  = "coding-upstream"
)

const promptCompressionDispatchConfigYAML = `
version: v0.3
routing:
  modelCards:
    - name: general-model
    - name: coding-model
  signals:
    keywords:
      - name: refactor_request
        operator: OR
        keywords: ["refactor"]
      - name: dropped_detail
        operator: OR
        keywords: ["zanzibar"]
  decisions:
    - name: compressed_view
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: refactor_request
          - operator: NOT
            conditions:
              - type: keyword
                name: dropped_detail
      modelRefs:
        - model: coding-model
          use_reasoning: false
providers:
  defaults:
    model: general-model
  models:
    - name: general-model
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
    - name: coding-model
      provider_model_id: ` + compressionRoutedUpstreamModel + `
      api_format: %s
      backend_refs:
        - endpoint: 127.0.0.1:8001
          provider: vllm
global:
  model_catalog:
    modules:
      prompt_compression:
        enabled: true
        max_tokens: 16
`

func TestPromptCompressionKeepsChatDispatchPromptAndTools(t *testing.T) {
	body := mustMarshalCompressionFixture(t, map[string]any{
		"model": config.DefaultVSRAutoModelName,
		"messages": []any{
			map[string]any{"role": "system", "content": compressionAgentInstructions},
			map[string]any{"role": "user", "content": compressionAgentPrompt},
		},
		"tools": []any{map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "apply_patch",
				"description": compressionAgentToolDescription,
				"parameters":  json.RawMessage(compressionAgentToolSchema),
			},
		}},
	})

	upstream := dispatchWithPromptCompression(t, "/v1/chat/completions", config.APIFormatOpenAI, body)
	assertUncompressedAgentDispatch(t, llmprotocol.OpenAIChatV1, upstream)
}

func TestPromptCompressionKeepsResponsesDispatchPromptAndTools(t *testing.T) {
	body := mustMarshalCompressionFixture(t, map[string]any{
		"model":        config.DefaultVSRAutoModelName,
		"instructions": compressionAgentInstructions,
		"input": []any{map[string]any{
			"type":    "message",
			"role":    "user",
			"content": []any{map[string]any{"type": "input_text", "text": compressionAgentPrompt}},
		}},
		"tools": []any{map[string]any{
			"type":        "function",
			"name":        "apply_patch",
			"description": compressionAgentToolDescription,
			"parameters":  json.RawMessage(compressionAgentToolSchema),
		}},
	})

	upstream := dispatchWithPromptCompression(t, "/v1/responses", config.APIFormatResponses, body)
	assertUncompressedAgentDispatch(t, llmprotocol.OpenAIResponsesV1, upstream)
}

func dispatchWithPromptCompression(t *testing.T, path string, apiFormat string, body []byte) []byte {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(promptCompressionDispatchConfigYAML, apiFormat)))
	require.NoError(t, err)
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	require.NoError(t, err)
	router := &OpenAIRouter{
		Config:             cfg,
		Classifier:         classifiers.Default(),
		RecipeClassifiers:  classifiers,
		Cache:              &spyCache{},
		CredentialResolver: newTestCredentialResolver(cfg),
	}

	ctx := &RequestContext{Headers: map[string]string{}, TraceContext: t.Context()}
	_, err = router.handleRequestHeaders(newRequestHeaders("POST", path), ctx)
	require.NoError(t, err)
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: body, EndOfStream: true},
	}, ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse(), "request ended before provider dispatch")
	upstream := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	require.NotEmpty(t, upstream, "ExtProc did not produce a provider request body")
	return upstream
}

func assertUncompressedAgentDispatch(t *testing.T, format llmprotocol.WireFormat, upstream []byte) {
	t.Helper()
	var routed struct {
		Model string `json:"model"`
	}
	require.NoError(t, json.Unmarshal(upstream, &routed))
	require.Equal(t, compressionRoutedUpstreamModel, routed.Model,
		"the decision did not see compressed text, so compression never ran for this request")
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(format, upstream)
	require.NoError(t, err, "provider body is not valid %s:\n%s", format, upstream)

	require.Len(t, request.Instructions, 1)
	assert.Equal(t, compressionAgentInstructions, semanticText(request.Instructions[0].Content))
	require.NotEmpty(t, request.Messages)
	current := request.Messages[len(request.Messages)-1]
	require.Equal(t, llmprotocol.RoleUser, current.Role)
	assert.Equal(t, compressionAgentPrompt, semanticText(current.Content),
		"the provider received a different prompt than the client sent")

	require.Len(t, request.Tools, 1, "the provider lost the client's tool definitions")
	assert.Equal(t, "apply_patch", request.Tools[0].Name)
	assert.Equal(t, compressionAgentToolDescription, request.Tools[0].Description)
	assert.JSONEq(t, compressionAgentToolSchema, string(request.Tools[0].InputSchema))
}

func mustMarshalCompressionFixture(t *testing.T, value map[string]any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	require.NoError(t, err)
	return body
}
