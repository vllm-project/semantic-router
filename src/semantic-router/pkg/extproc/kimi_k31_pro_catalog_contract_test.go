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

const kimiK31ProCatalogID = "moonshot/kimi-k3.1-pro"

func TestKimiK31ProReasoningWireContracts(t *testing.T) {
	for _, effort := range []string{"", "low", "high", "max"} {
		name, selected := effort, effort
		if effort == "" {
			name, selected = "default", "max"
		}
		runCatalogReasoningWireCases(t, []catalogReasoningWireCase{{
			name: name, catalog: kimiK31ProCatalogID, provider: "moonshot",
			enabled: true, mode: config.ReasoningModeEnabled, effort: effort,
			wantTransport: modelcatalog.ReasoningTransportTopLevelEffort,
			wantControls:  map[string]interface{}{"reasoning_effort": selected},
		}})
	}
}

func TestKimiK31ProRejectsUnsupportedReasoning(t *testing.T) {
	for _, test := range []struct {
		name    string
		control string
		wantErr string
	}{
		{"disabled", "use_reasoning: false", "always-on reasoning family"},
		{"medium effort", "use_reasoning: true\n            reasoning_effort: medium", `reasoning_effort "medium" is not supported`},
		{"adaptive mode", "use_reasoning: true\n            reasoning_mode: adaptive", `reasoning_mode "adaptive" is not supported`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := config.ParseYAMLBytes(kimiK31ProConfig("http://127.0.0.1:8000/v1", test.control))
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

// Exercise the catalog-backed alias through routing and the final HTTP hop:
// the Moonshot-native model ID, bearer auth, reasoning effort, and the complete
// assistant turn that Kimi K3.1 Pro requires on tool calls.
func TestKimiK31ProProviderDispatch(t *testing.T) {
	type observation struct {
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
		received <- observation{r.URL.Path, r.Header.Get("Authorization"), body}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(fixture.Close)
	cfg, err := config.ParseYAMLBytes(kimiK31ProConfig(fixture.URL+"/v1",
		"use_reasoning: true\n            reasoning_effort: high"))
	require.NoError(t, err)
	decision := cfg.GetDecisionByName("kimi_default")
	require.NotNil(t, decision)
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
	}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, nil)
	ctx.VSRSelectedDecision = decision
	input := []byte(`{"model":"kimi","messages":[{"role":"user","content":"What is the weather?"},{"role":"assistant","content":null,"reasoning_content":"I should check the weather.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"Sunny"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{}}}}]}`)
	request, immediate := router.prepareProtocolRequest(input, ctx)
	require.Nil(t, immediate)
	response, err := router.handleEntrypointModelRouting(request, "kimi", decision.Name,
		entropy.ReasoningDecision{UseReasoning: true}, "kimi", ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	common := response.GetRequestBody().GetResponse()
	require.NotNil(t, common)
	emitted := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	dispatch, err := router.resolveProviderDispatch("kimi", decision.Name, true)
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
	assert.Equal(t, "/v1/chat/completions", observed.path)
	assert.Equal(t, "Bearer moonshot-fixture-key", observed.authorization)
	wire := unmarshalReasoningRequest(t, observed.body)
	assert.Equal(t, "kimi-k3.1-pro", wire["model"])
	assert.Equal(t, map[string]interface{}{"reasoning_effort": "high"}, providerReasoningControls(wire))
	original := unmarshalReasoningRequest(t, input)
	assert.Equal(t, original["tools"], wire["tools"])
	// The Chat codec omits optional null assistant content.
	delete(original["messages"].([]interface{})[1].(map[string]interface{}), "content")
	assert.Equal(t, original["messages"], wire["messages"])
}

func kimiK31ProConfig(endpoint, control string) []byte {
	return []byte(fmt.Sprintf(`version: v0.5
providers:
  models:
    - name: kimi
      catalog: %s
      backend_refs:
        - name: primary
          provider: moonshot
          endpoint: %s
          api_key: moonshot-fixture-key
recipes:
  - name: default
    decisions:
      - name: kimi_default
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: kimi
            %s
`, kimiK31ProCatalogID, endpoint, control))
}
