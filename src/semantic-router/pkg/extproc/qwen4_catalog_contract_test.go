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
// uses top_level_effort_boolean_switch: an enabled decision sends
// reasoning_effort (default high) and enable_thinking true. A disabled
// decision sends enable_thinking false and leaves reasoning_effort off.
func TestQwen4MaxUsesWorkspaceEndpointAndDeclaredEfforts(t *testing.T) {
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
			control := "use_reasoning: true"
			if effort != "" {
				control += "\n            reasoning_effort: " + effort
			}
			wire := dispatchQwen4Max(t, control, true)
			assert.Equal(t, map[string]interface{}{
				"reasoning_effort": selected,
				"enable_thinking":  true,
			}, providerReasoningControls(wire))
		})
	}
}

func TestQwen4MaxDisablesThinking(t *testing.T) {
	for _, test := range []struct {
		name    string
		control string
	}{
		{name: "use_reasoning false", control: "use_reasoning: false"},
		{
			name:    "reasoning_mode disabled",
			control: "use_reasoning: false\n            reasoning_mode: disabled",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			wire := dispatchQwen4Max(t, test.control, false)
			assert.Equal(t, map[string]interface{}{
				"enable_thinking": false,
			}, providerReasoningControls(wire))
			assert.NotContains(t, wire, "reasoning_effort")
		})
	}
}

func TestQwen4MaxRejectsUndeclaredReasoning(t *testing.T) {
	for _, effort := range []string{"none", "minimal", "medium", "max"} {
		t.Run(effort, func(t *testing.T) {
			_, err := config.ParseYAMLBytes(qwen4MaxConfig("http://127.0.0.1:8000/compatible-mode/v1",
				"use_reasoning: true\n            reasoning_effort: "+effort))
			require.ErrorContains(t, err, fmt.Sprintf("reasoning_effort %q is not supported", effort))
		})
	}
	_, err := config.ParseYAMLBytes(qwen4MaxConfig("http://127.0.0.1:8000/compatible-mode/v1",
		"use_reasoning: false\n            reasoning_effort: low"))
	require.ErrorContains(t, err, "reasoning_effort cannot be set while reasoning is disabled")
	_, err = config.ParseYAMLBytes(qwen4MaxConfig("http://127.0.0.1:8000/compatible-mode/v1",
		"use_reasoning: true\n            reasoning_mode: "+config.ReasoningModeDisabled))
	require.ErrorContains(t, err, fmt.Sprintf("reasoning_mode %q", config.ReasoningModeDisabled))
	_, err = config.ParseYAMLBytes(qwen4MaxConfig("http://127.0.0.1:8000/compatible-mode/v1",
		"use_reasoning: true\n            reasoning_mode: "+config.ReasoningModeAdaptive))
	require.ErrorContains(t, err, fmt.Sprintf("reasoning_mode %q is not supported", config.ReasoningModeAdaptive))
}

func dispatchQwen4Max(t *testing.T, control string, useReasoning bool) map[string]interface{} {
	t.Helper()
	type observation struct {
		method        string
		path          string
		authorization string
		body          []byte
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
		entropy.ReasoningDecision{UseReasoning: useReasoning}, "routed", ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	common := response.GetRequestBody().GetResponse()
	require.NotNil(t, common)
	emitted := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	dispatch, err := router.resolveProviderDispatch("routed", decision.Name, useReasoning)
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
		return wire
	default:
		t.Fatal("DashScope backend did not receive the request")
		return nil
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
