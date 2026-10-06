package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/internalauth"
)

func TestLooperReasoningSurvivesProviderBoundary(t *testing.T) {
	for _, test := range []struct {
		name       string
		recipe     string
		decision   string
		wantMode   *bool
		wantPrompt bool
	}{
		{"disabled worker", "quiet", "shared", boolPtr(false), true},
		{"enabled worker", "thinking", "shared", boolPtr(true), true},
		{"no decision", "quiet", "", nil, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte(looperReasoningWireConfig))
			require.NoError(t, err)
			require.Equal(t, "qwen3", cfg.ModelConfig["worker-alias"].ReasoningFamily)
			router := &OpenAIRouter{
				Config: cfg, Cache: &spyCache{}, CredentialResolver: newTestCredentialResolver(cfg),
			}
			ctx := &RequestContext{Headers: map[string]string{}}
			requestHeaders := newRequestHeaders("POST", "/v1/chat/completions")
			for key, value := range map[string]string{
				headers.VSRLooperRequest: "true", headers.VSRInternalAuth: internalauth.Token(),
				headers.VSRLooperDecision: test.decision, headers.VSRSelectedRecipe: test.recipe,
			} {
				requestHeaders.RequestHeaders.Headers.Headers = append(
					requestHeaders.RequestHeaders.Headers.Headers, &core.HeaderValue{Key: key, Value: value},
				)
			}
			_, err = router.handleRequestHeaders(requestHeaders, ctx)
			require.NoError(t, err)
			require.True(t, ctx.LooperRequest)
			response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
				RequestBody: &ext_proc.HttpBody{Body: []byte(`{
					"model":"worker-alias","messages":[
						{"role":"system","content":"Original instruction."},
						{"role":"user","content":"Plan the requested work."}
					],"max_completion_tokens":4096,"response_format":{"type":"json_object"}
				}`)},
			}, ctx)
			require.NoError(t, err)
			require.NotNil(t, response.GetRequestBody())
			common := response.GetRequestBody().GetResponse()
			body := common.GetBodyMutation().GetBody()
			var wire map[string]interface{}
			require.NoError(t, json.Unmarshal(body, &wire))
			assert.Equal(t, "physical-worker", wire["model"])
			assert.Equal(t, float64(4096), wire["max_completion_tokens"])
			assert.Equal(t, map[string]interface{}{"type": "json_object"}, wire["response_format"])
			if test.wantMode == nil {
				assert.NotContains(t, wire, "chat_template_kwargs")
			} else {
				assert.Equal(t, map[string]interface{}{"enable_thinking": *test.wantMode}, wire["chat_template_kwargs"])
				assert.Equal(t, test.recipe, string(ctx.Routing.RecipeName()))
			}
			wantPromptCount := 0
			if test.wantPrompt {
				wantPromptCount = 1
			}
			assert.Equal(t, wantPromptCount, strings.Count(string(body), "Stage instruction."), "insert the decision prompt once")
			assert.Contains(t, string(body), "Original instruction.")
			for _, header := range common.GetHeaderMutation().GetSetHeaders() {
				assert.NotEqual(t, headers.VSRInternalAuth, header.GetHeader().GetKey())
			}
		})
	}
}

var looperReasoningWireConfig = `
version: v0.3
providers:
  models:
    - name: worker-alias
      provider_model_id: physical-worker
      reasoning:
        family: qwen3
      backend_refs:
        - provider: vllm
          endpoint: 127.0.0.1:8000
routing: {}
recipes:
` + looperReasoningWireRecipe("quiet", false) + looperReasoningWireRecipe("thinking", true)

func looperReasoningWireRecipe(name string, enabled bool) string {
	return fmt.Sprintf(`
  - name: %s
    routing:
      decisions:
        - name: shared
          rules:
            operator: AND
            conditions: []
          modelRefs:
            - model: worker-alias
              use_reasoning: %t
          plugins:
            - type: system_prompt
              configuration:
                system_prompt: "Stage instruction."
                mode: insert
`, name, enabled)
}
