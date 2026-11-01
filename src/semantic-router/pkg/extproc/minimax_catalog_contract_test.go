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

const miniMaxFlashCatalogID = "minimax/minimax-m3.1-flash"

func TestMiniMaxFlashReasoningWireContracts(t *testing.T) {
	for _, provider := range []string{"minimax", "vllm"} {
		for _, effort := range []string{"", "low", "medium", "high", "xhigh", "max"} {
			selected := effort
			if selected == "" {
				selected = "max"
			}
			transport := modelcatalog.ReasoningTransportTopLevelEffort
			if provider == "vllm" {
				transport = modelcatalog.ReasoningTransportChatTemplate
			}
			runCatalogReasoningWireCases(t, []catalogReasoningWireCase{{
				name: provider + "/" + effort, catalog: miniMaxFlashCatalogID, provider: provider,
				enabled: true, mode: config.ReasoningModeAdaptive, effort: effort,
				wantTransport: transport, wantControls: miniMaxFlashControls(provider, selected),
			}})
		}
	}
}

func TestMiniMaxFlashRejectsUnsupportedReasoning(t *testing.T) {
	for _, provider := range []string{"minimax", "vllm"} {
		for _, test := range []struct {
			name    string
			control string
			wantErr string
		}{
			{"disabled", "use_reasoning: false", "always-on reasoning family"},
			{"none effort", "use_reasoning: true\n          reasoning_effort: none", `reasoning_effort "none" is not supported`},
			{"enabled mode", "use_reasoning: true\n          reasoning_mode: enabled", `reasoning_mode "enabled" is not supported`},
			{"unknown effort", "use_reasoning: true\n          reasoning_effort: minimal", `reasoning_effort "minimal" is not supported`},
		} {
			t.Run(provider+"/"+test.name, func(t *testing.T) {
				_, err := config.ParseYAMLBytes(miniMaxFlashConfig(provider, "http://127.0.0.1:8000/v1", test.control))
				require.ErrorContains(t, err, test.wantErr)
			})
		}
	}
}

// Exercise the catalog-backed alias through routing and the final HTTP hop,
// including provider-native model IDs, auth, tool history, and reasoning depth.
func TestMiniMaxFlashProviderDispatch(t *testing.T) {
	for _, provider := range []string{"minimax", "vllm"} {
		t.Run(provider, func(t *testing.T) {
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
			cfg, err := config.ParseYAMLBytes(miniMaxFlashConfig(provider, fixture.URL+"/v1",
				"use_reasoning: true\n          reasoning_effort: high"))
			require.NoError(t, err)
			decision := cfg.GetDecisionByName("flash_default")
			require.NotNil(t, decision)
			router := &OpenAIRouter{
				Config:             cfg,
				CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
			}
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, nil)
			ctx.VSRSelectedDecision = decision
			input := []byte(`{"model":"flash","messages":[{"role":"user","content":"What is the weather?"},{"role":"assistant","content":null,"reasoning_content":"I should check the weather.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"Sunny"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{}}}}]}`)
			request, immediate := router.prepareProtocolRequest(input, ctx)
			require.Nil(t, immediate)
			response, err := router.handleEntrypointModelRouting(request, "flash", decision.Name,
				entropy.ReasoningDecision{UseReasoning: true}, "flash", ctx)
			require.NoError(t, err)
			require.Nil(t, response.GetImmediateResponse())
			common := response.GetRequestBody().GetResponse()
			require.NotNil(t, common)
			emitted := headerValuesByName(common.GetHeaderMutation().GetSetHeaders())
			dispatch, err := router.resolveProviderDispatch("flash", decision.Name, true)
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
			assert.Equal(t, "Bearer minimax-fixture-key", observed.authorization)
			wire := unmarshalReasoningRequest(t, observed.body)
			wantModel := "MiniMax-M3.1-Flash"
			if provider == "vllm" {
				wantModel = "MiniMaxAI/MiniMax-M3.1-Flash"
			}
			assert.Equal(t, wantModel, wire["model"])
			assert.Equal(t, miniMaxFlashControls(provider, "high"), providerReasoningControls(wire))
			original := unmarshalReasoningRequest(t, input)
			assert.Equal(t, original["tools"], wire["tools"])
			// The Chat codec omits optional null assistant content.
			delete(original["messages"].([]interface{})[1].(map[string]interface{}), "content")
			assert.Equal(t, original["messages"], wire["messages"])
		})
	}
}

func miniMaxFlashControls(provider, effort string) map[string]interface{} {
	controls := map[string]interface{}{"reasoning_effort": effort}
	if provider == "vllm" {
		return map[string]interface{}{"chat_template_kwargs": controls}
	}
	return controls
}

func miniMaxFlashConfig(provider, endpoint, control string) []byte {
	return []byte(fmt.Sprintf(`version: v0.3
providers:
  models:
    - name: flash
      catalog: %s
      backend_refs:
        - name: primary
          provider: %s
          base_url: %s
          api_key: minimax-fixture-key
routing:
  decisions:
    - name: flash_default
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: flash
          %s
`, miniMaxFlashCatalogID, provider, endpoint, control))
}
