package extproc

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// A workspace host replaces the legacy DashScope default. Chat Completions
// keeps the provider's top-level reasoning_effort, including the declared
// default of high when the decision does not name an effort.
func TestQwen4MaxUsesWorkspaceEndpointAndDeclaredEfforts(t *testing.T) {
	type observation struct {
		method        string
		path          string
		authorization string
		body          []byte
	}
	for _, effort := range []string{"", "low", "high", "xhigh"} {
		name := effort
		if name == "" {
			name = "default"
		}
		t.Run(name, func(t *testing.T) {
			selected := effort
			if selected == "" {
				selected = "high"
			}
			received := make(chan observation, 1)
			fixture := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				received <- observation{
					method: r.Method, path: r.URL.Path,
					authorization: r.Header.Get("Authorization"), body: body,
				}
				w.WriteHeader(http.StatusOK)
			}))
			t.Cleanup(fixture.Close)
			control := "use_reasoning: true"
			if effort != "" {
				control += "\n            reasoning_effort: " + effort
			}
			endpoint := fixture.URL + "/compatible-mode/v1"
			cfg, err := config.ParseYAMLBytes(qwen4MaxConfig(endpoint, control))
			require.NoError(t, err)
			decision := cfg.GetDecisionByName("qwen4_default")
			require.NotNil(t, decision)
			router := &OpenAIRouter{
				Config:             cfg,
				CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
			}
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, nil)
			ctx.VSRSelectedDecision = decision
			request, immediate := router.prepareProtocolRequest(
				[]byte(`{"model":"routed","messages":[{"role":"user","content":"hello"}]}`), ctx)
			require.Nil(t, immediate)
			response, err := router.handleEntrypointModelRouting(request, "routed", decision.Name,
				entropy.ReasoningDecision{UseReasoning: true}, "routed", ctx)
			require.NoError(t, err)
			require.Nil(t, response.GetImmediateResponse())
			common := response.GetRequestBody().GetResponse()
			require.NotNil(t, common)
			emitted := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
			dispatch, err := router.resolveProviderDispatch("routed", decision.Name, true)
			require.NoError(t, err)
			baseURL := providerEndpointScheme(cfg, dispatch.backendName, dispatch.profile) + "://" + dispatch.backendAddress
			require.Equal(t, fixture.URL, baseURL)
			require.NotContains(t, baseURL, "dashscope-intl.aliyuncs.com")
			outbound, err := http.NewRequest(http.MethodPost, baseURL+emitted[":path"], bytes.NewReader(common.GetBodyMutation().GetBody()))
			require.NoError(t, err)
			for headerName, value := range emitted {
				if !strings.HasPrefix(headerName, ":") {
					outbound.Header.Set(headerName, value)
				}
			}
			result, err := fixture.Client().Do(outbound)
			require.NoError(t, err)
			require.NoError(t, result.Body.Close())
			require.Equal(t, http.StatusOK, result.StatusCode)
			select {
			case observed := <-received:
				assert.Equal(t, http.MethodPost, observed.method)
				assert.Equal(t, "/compatible-mode/v1/chat/completions", observed.path)
				assert.Equal(t, "Bearer dashscope-fixture-key", observed.authorization)
				wire := unmarshalReasoningRequest(t, observed.body)
				assert.Equal(t, "qwen4-max", wire["model"])
				assert.Equal(t, map[string]interface{}{"reasoning_effort": selected}, providerReasoningControls(wire))
				assert.NotContains(t, wire, "enable_thinking")
			default:
				t.Fatal("DashScope backend did not receive the request")
			}
		})
	}
}

func TestQwen4MaxRejectsUndeclaredReasoning(t *testing.T) {
	_, err := config.ParseYAMLBytes(qwen4MaxConfig("http://127.0.0.1:8000/compatible-mode/v1", "use_reasoning: false"))
	require.ErrorContains(t, err, "always-on reasoning family")
	for _, effort := range []string{"none", "minimal", "medium", "max"} {
		t.Run(effort, func(t *testing.T) {
			_, err := config.ParseYAMLBytes(qwen4MaxConfig("http://127.0.0.1:8000/compatible-mode/v1",
				"use_reasoning: true\n            reasoning_effort: "+effort))
			require.ErrorContains(t, err, fmt.Sprintf("reasoning_effort %q is not supported", effort))
		})
	}
	for _, mode := range []string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled} {
		t.Run("mode/"+mode, func(t *testing.T) {
			_, err := config.ParseYAMLBytes(qwen4MaxConfig("http://127.0.0.1:8000/compatible-mode/v1",
				"use_reasoning: true\n            reasoning_mode: "+mode))
			require.ErrorContains(t, err, fmt.Sprintf("reasoning_mode %q is not supported", mode))
		})
	}
}

func qwen4MaxConfig(endpoint, control string) []byte {
	return []byte(fmt.Sprintf(`version: v0.5
providers:
  models:
    - name: routed
      catalog: qwen/qwen4-max
      backend_refs:
        - name: primary
          provider: dashscope
          endpoint: %s
          api_key: dashscope-fixture-key
recipes:
  - name: default
    decisions:
      - name: qwen4_default
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: routed
            %s
`, endpoint, control))
}
