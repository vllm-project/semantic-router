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

const geminiFlashLiteCatalogID = "google/gemini-3.5-flash-lite"

// Every documented thinking level reaches the Gemini API as a top-level
// reasoning_effort, and an omitted effort uses the documented minimal default.
func TestGeminiFlashLiteReasoningWireContracts(t *testing.T) {
	for _, effort := range []string{"", "minimal", "low", "medium", "high"} {
		selected := effort
		if selected == "" {
			selected = "minimal"
		}
		runCatalogReasoningWireCases(t, []catalogReasoningWireCase{{
			name: "gemini/" + effort, catalog: geminiFlashLiteCatalogID, provider: "gemini",
			enabled: true, mode: config.ReasoningModeEnabled, effort: effort,
			wantTransport: modelcatalog.ReasoningTransportTopLevelEffort,
			wantControls:  map[string]interface{}{"reasoning_effort": selected},
		}})
	}
}

// Thinking cannot be turned off, and levels outside the documented ladder are
// rejected when the configuration loads instead of failing at the provider.
func TestGeminiFlashLiteRejectsUnsupportedReasoning(t *testing.T) {
	for _, test := range []struct {
		name    string
		control string
		wantErr string
	}{
		{"disabled", "use_reasoning: false", "always-on reasoning family"},
		{"xhigh effort", "use_reasoning: true\n            reasoning_effort: xhigh", `reasoning_effort "xhigh" is not supported`},
		{"max effort", "use_reasoning: true\n            reasoning_effort: max", `reasoning_effort "max" is not supported`},
		{"none effort", "use_reasoning: true\n            reasoning_effort: none", `reasoning_effort "none" is not supported`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := config.ParseYAMLBytes(geminiFlashLiteConfig("https://generativelanguage.googleapis.com/v1beta/openai", test.control))
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}

// Exercise the catalog-backed alias through routing and the final HTTP hop:
// the native model ID, the Chat Completions path, auth, and reasoning depth.
func TestGeminiFlashLiteProviderDispatch(t *testing.T) {
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
	cfg, err := config.ParseYAMLBytes(geminiFlashLiteConfig(fixture.URL+"/v1beta/openai",
		"use_reasoning: true\n            reasoning_effort: high"))
	require.NoError(t, err)
	decision := cfg.GetDecisionByName("flash_lite_default")
	require.NotNil(t, decision)
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
	}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, nil)
	ctx.VSRSelectedDecision = decision
	input := []byte(`{"model":"flash-lite","messages":[{"role":"user","content":"Summarize this page."}]}`)
	request, immediate := router.prepareProtocolRequest(input, ctx)
	require.Nil(t, immediate)
	response, err := router.handleEntrypointModelRouting(request, "flash-lite", decision.Name,
		entropy.ReasoningDecision{UseReasoning: true}, "flash-lite", ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	common := response.GetRequestBody().GetResponse()
	require.NotNil(t, common)
	emitted := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
	dispatch, err := router.resolveProviderDispatch("flash-lite", decision.Name, true)
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
	assert.Equal(t, "/v1beta/openai/chat/completions", observed.path)
	assert.Equal(t, "Bearer gemini-fixture-key", observed.authorization)
	wire := unmarshalReasoningRequest(t, observed.body)
	assert.Equal(t, "gemini-3.5-flash-lite", wire["model"])
	assert.Equal(t, map[string]interface{}{"reasoning_effort": "high"}, providerReasoningControls(wire))
	original := unmarshalReasoningRequest(t, input)
	assert.Equal(t, original["messages"], wire["messages"])
}

func geminiFlashLiteConfig(endpoint, control string) []byte {
	return []byte(fmt.Sprintf(`version: v0.5
providers:
  models:
    - name: flash-lite
      catalog: %s
      backend_refs:
        - name: primary
          provider: gemini
          endpoint: %s
          api_key: gemini-fixture-key
recipes:
  - name: default
    decisions:
      - name: flash_lite_default
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: flash-lite
            %s
`, geminiFlashLiteCatalogID, endpoint, control))
}
