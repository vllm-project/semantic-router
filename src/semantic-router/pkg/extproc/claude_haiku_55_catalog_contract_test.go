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
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

const claudeHaiku55CatalogID = "anthropic/claude-haiku-5.5"

// Claude Haiku 5.5 takes all five effort levels in output_config.effort with
// adaptive thinking, and medium when the route names no effort.
func TestClaudeHaiku55ReasoningWireContracts(t *testing.T) {
	for _, effort := range []string{"", "low", "medium", "high", "xhigh", "max"} {
		name, selected := effort, effort
		if effort == "" {
			name, selected = "default", "medium"
		}
		runCatalogReasoningWireCases(t, []catalogReasoningWireCase{{
			name: name, catalog: claudeHaiku55CatalogID, provider: "anthropic",
			enabled: true, mode: config.ReasoningModeAdaptive, effort: effort,
			wantTransport: modelcatalog.ReasoningTransportOutputConfig,
			wantControls: map[string]interface{}{
				"thinking":      map[string]interface{}{"type": "adaptive"},
				"output_config": map[string]interface{}{"effort": selected},
			},
		}})
	}
	runCatalogReasoningWireCases(t, []catalogReasoningWireCase{{
		name: "disabled", catalog: claudeHaiku55CatalogID, provider: "anthropic",
		enabled: false, mode: config.ReasoningModeDisabled,
		wantTransport: modelcatalog.ReasoningTransportOutputConfig,
		wantControls:  map[string]interface{}{"thinking": map[string]interface{}{"type": "disabled"}},
	}})
}

func TestClaudeHaiku55RejectsUnsupportedReasoning(t *testing.T) {
	for _, test := range []struct {
		name    string
		control string
		wantErr string
	}{
		{"minimal effort", "use_reasoning: true\n            reasoning_effort: minimal", `reasoning_effort "minimal" is not supported`},
		{"enabled mode", "use_reasoning: true\n            reasoning_mode: enabled", `reasoning_mode "enabled" is not supported`},
		// Thinking can be turned off only at high effort or below, so a disabled
		// route must not carry an effort of its own.
		{"disabled with an effort", "use_reasoning: false\n            reasoning_effort: max", "reasoning_effort cannot be set while reasoning is disabled"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := config.ParseYAMLBytes(claudeHaiku55Config("https://api.anthropic.com", test.control))
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

// Exercise the catalog-backed alias through routing and the final HTTP hop:
// the Anthropic-native model ID, the x-api-key credential, the Messages
// version header, and adaptive thinking with the routed effort.
func TestClaudeHaiku55ProviderDispatch(t *testing.T) {
	type observation struct {
		path    string
		apiKey  string
		version string
		body    []byte
	}
	received := make(chan observation, 1)
	fixture := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		received <- observation{r.URL.Path, r.Header.Get("x-api-key"), r.Header.Get("anthropic-version"), body}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(fixture.Close)
	cfg, err := config.ParseYAMLBytes(claudeHaiku55Config(fixture.URL,
		"use_reasoning: true\n            reasoning_effort: max"))
	require.NoError(t, err)
	decision := cfg.GetDecisionByName("haiku_default")
	require.NotNil(t, decision)
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
	}
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, nil)
	ctx.VSRSelectedDecision = decision
	input := []byte(`{"model":"haiku","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"Classify this ticket."}]}]}`)
	request, immediate := router.prepareProtocolRequest(input, ctx)
	require.Nil(t, immediate)
	response, err := router.handleEntrypointModelRouting(request, "haiku", decision.Name,
		entropy.ReasoningDecision{UseReasoning: true}, "haiku", ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	common := response.GetRequestBody().GetResponse()
	require.NotNil(t, common)
	emitted := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	dispatch, err := router.resolveProviderDispatch("haiku", decision.Name, true)
	require.NoError(t, err)
	baseURL := providerEndpointScheme(cfg, dispatch.backendName, dispatch.profile) + "://" + dispatch.backendAddress
	require.Equal(t, fixture.URL, baseURL)
	outbound, err := http.NewRequest(http.MethodPost, baseURL+emitted[":path"], bytes.NewReader(common.GetBodyMutation().GetBody()))
	require.NoError(t, err)
	for name, value := range emitted {
		if !strings.HasPrefix(name, ":") {
			outbound.Header.Set(name, value)
		}
	}
	result, err := fixture.Client().Do(outbound)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, result.Body.Close()) })
	require.Equal(t, http.StatusOK, result.StatusCode)
	observed := <-received
	assert.Equal(t, "/v1/messages", observed.path)
	assert.Equal(t, "anthropic-fixture-key", observed.apiKey)
	assert.Equal(t, "2023-06-01", observed.version)
	wire := unmarshalReasoningRequest(t, observed.body)
	assert.Equal(t, "claude-haiku-5-5", wire["model"])
	assert.Equal(t, map[string]interface{}{
		"thinking":      map[string]interface{}{"type": "adaptive"},
		"output_config": map[string]interface{}{"effort": "max"},
	}, providerReasoningControls(wire))
	assert.Equal(t, unmarshalReasoningRequest(t, input)["messages"], wire["messages"])
}

func claudeHaiku55Config(endpoint, control string) []byte {
	return []byte(fmt.Sprintf(`version: v0.5
providers:
  models:
    - name: haiku
      catalog: %s
      backend_refs:
        - name: primary
          provider: anthropic
          endpoint: %s
          api_key: anthropic-fixture-key
recipes:
  - name: default
    decisions:
      - name: haiku_default
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: haiku
            %s
`, claudeHaiku55CatalogID, endpoint, control))
}
