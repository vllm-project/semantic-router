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

const (
	longCat25CatalogID = "meituan/longcat-2.5"
	longCat20CatalogID = "meituan/longcat-2.0"

	longCatFlashLiteSparseCatalogID = "meituan/longcat-flash-lite-sparse"
)

// The LongCat API turns LongCat-2.5 thinking on or off with a thinking object
// and takes no effort.
func TestLongCat25ReasoningWireContracts(t *testing.T) {
	runCatalogReasoningWireCases(t, []catalogReasoningWireCase{
		{
			name: "enabled", catalog: longCat25CatalogID, provider: "longcat",
			enabled: true, mode: config.ReasoningModeEnabled,
			wantTransport: modelcatalog.ReasoningTransportThinkingObject,
			wantControls:  map[string]interface{}{"thinking": map[string]interface{}{"type": "enabled"}},
		},
		{
			name: "enabled without a mode", catalog: longCat25CatalogID, provider: "longcat",
			enabled:       true,
			wantTransport: modelcatalog.ReasoningTransportThinkingObject,
			wantControls:  map[string]interface{}{"thinking": map[string]interface{}{"type": "enabled"}},
		},
		{
			name: "disabled", catalog: longCat25CatalogID, provider: "longcat",
			enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportThinkingObject,
			wantControls:  map[string]interface{}{"thinking": map[string]interface{}{"type": "disabled"}},
		},
	})
}

// The LongCat-2.0 weights take their thinking switch as a chat template
// argument, which is how an SGLang backend receives it.
func TestLongCat20SGLangReasoningWireContracts(t *testing.T) {
	templateSwitch := func(enabled bool) map[string]interface{} {
		return map[string]interface{}{"chat_template_kwargs": map[string]interface{}{"enable_thinking": enabled}}
	}
	runCatalogReasoningWireCases(t, []catalogReasoningWireCase{
		{
			name: "enabled", catalog: longCat20CatalogID, provider: "sglang",
			enabled: true, mode: config.ReasoningModeEnabled,
			wantTransport: modelcatalog.ReasoningTransportChatTemplate, wantControls: templateSwitch(true),
		},
		{
			name: "enabled without a mode", catalog: longCat20CatalogID, provider: "sglang",
			enabled:       true,
			wantTransport: modelcatalog.ReasoningTransportChatTemplate, wantControls: templateSwitch(true),
		},
		{
			name: "disabled", catalog: longCat20CatalogID, provider: "sglang",
			enabled: false, mode: config.ReasoningModeDisabled,
			wantTransport: modelcatalog.ReasoningTransportChatTemplate, wantControls: templateSwitch(false),
		},
	})
}

// LongCat-Flash-Lite-Sparse is a non-thinking model with no reasoning family,
// so its SGLang backend receives no thinking switch.
func TestLongCatFlashLiteSparseSendsNoReasoningControl(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		runCatalogReasoningWireCases(t, []catalogReasoningWireCase{{
			name: fmt.Sprintf("use_reasoning %t", enabled), catalog: longCatFlashLiteSparseCatalogID,
			provider: "sglang", enabled: enabled,
			wantTransport: modelcatalog.ReasoningTransportChatTemplate,
			wantControls:  map[string]interface{}{},
		}})
	}
}

func TestLongCat25RejectsUnsupportedReasoning(t *testing.T) {
	for _, test := range []struct {
		name    string
		control string
		wantErr string
	}{
		{"an effort", "use_reasoning: true\n            reasoning_effort: high", "reasoning_effort cannot be used with this mode-only reasoning family"},
		{"adaptive mode", "use_reasoning: true\n            reasoning_mode: adaptive", `reasoning_mode "adaptive" is not supported`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := config.ParseYAMLBytes(longCat25Config("https://api.longcat.chat/openai/v1", test.control))
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

// A client's output limit reaches the LongCat API as max_tokens whichever
// protocol the client spoke, since the Chat codec renders it as
// max_completion_tokens.
func TestLongCatDispatchCarriesOutputLimitAsMaxTokens(t *testing.T) {
	for _, client := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(client), func(t *testing.T) {
			fields := dispatchBody(t, "longcat", client)
			assert.Equal(t, "8", string(fields["max_tokens"]))
			assert.NotContains(t, fields, "max_completion_tokens")
		})
	}
}

// Exercise the catalog-backed alias through routing and the final HTTP hop:
// the LongCat-native model ID, bearer auth, the /openai/v1 base path, the
// thinking object, and the output limit as max_tokens.
func TestLongCat25ProviderDispatch(t *testing.T) {
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
	cfg, err := config.ParseYAMLBytes(longCat25Config(fixture.URL+"/openai/v1", "use_reasoning: true"))
	require.NoError(t, err)
	decision := cfg.GetDecisionByName("longcat_default")
	require.NotNil(t, decision)
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
	}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, nil)
	ctx.VSRSelectedDecision = decision
	input := []byte(`{"model":"longcat","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello!"}],"max_tokens":32}`)
	request, immediate := router.prepareProtocolRequest(input, ctx)
	require.Nil(t, immediate)
	response, err := router.handleEntrypointModelRouting(request, "longcat", decision.Name,
		entropy.ReasoningDecision{UseReasoning: true}, "longcat", ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	common := response.GetRequestBody().GetResponse()
	require.NotNil(t, common)
	emitted := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	dispatch, err := router.resolveProviderDispatch("longcat", decision.Name, true)
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
	assert.Equal(t, "/openai/v1/chat/completions", observed.path)
	assert.Equal(t, "Bearer longcat-fixture-key", observed.authorization)
	wire := unmarshalReasoningRequest(t, observed.body)
	assert.Equal(t, "LongCat-2.5", wire["model"])
	assert.Equal(t, map[string]interface{}{
		"thinking": map[string]interface{}{"type": "enabled"},
	}, providerReasoningControls(wire))
	assert.Equal(t, unmarshalReasoningRequest(t, input)["messages"], wire["messages"])
	// The LongCat API documents max_tokens as its only output limit.
	assert.EqualValues(t, 32, wire["max_tokens"])
	assert.NotContains(t, wire, "max_completion_tokens")
}

func longCat25Config(endpoint, control string) []byte {
	return longCatConfig(longCat25CatalogID, "longcat", endpoint, control)
}

func longCatConfig(catalogID, provider, endpoint, control string) []byte {
	return []byte(fmt.Sprintf(`version: v0.5
providers:
  models:
    - name: longcat
      catalog: %s
      backend_refs:
        - name: primary
          provider: %s
          endpoint: %s
          api_key: longcat-fixture-key
recipes:
  - name: default
    decisions:
      - name: longcat_default
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: longcat
            %s
`, catalogID, provider, endpoint, control))
}
