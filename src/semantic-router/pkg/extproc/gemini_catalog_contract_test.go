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

// Exercise every supported effort and both defaults through the catalog-backed
// alias, routing, provider mapping, and final Chat Completions HTTP request.
func TestGemini4ProviderDispatch(t *testing.T) {
	type observation struct {
		method        string
		path          string
		authorization string
		body          []byte
	}
	for _, model := range []string{"argon", "carbon", "hydrogen", "neon"} {
		efforts := []string{"", "low", "medium", "high"}
		defaultEffort := "medium"
		if model == "hydrogen" || model == "neon" {
			efforts = append(efforts, "xhigh")
			defaultEffort = "high"
		}
		for _, effort := range efforts {
			name := effort
			if name == "" {
				name = "default"
			}
			t.Run(model+"/"+name, func(t *testing.T) {
				selected := effort
				if selected == "" {
					selected = defaultEffort
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
				cfg, err := config.ParseYAMLBytes(gemini4Config(model, fixture.URL+"/v1beta/openai", control))
				require.NoError(t, err)
				decision := cfg.GetDecisionByName("gemini_default")
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
				outbound, err := http.NewRequest(http.MethodPost, baseURL+emitted[":path"], bytes.NewReader(common.GetBodyMutation().GetBody()))
				require.NoError(t, err)
				for name, value := range emitted {
					if !strings.HasPrefix(name, ":") {
						outbound.Header.Set(name, value)
					}
				}
				result, err := fixture.Client().Do(outbound)
				require.NoError(t, err)
				require.NoError(t, result.Body.Close())
				require.Equal(t, http.StatusOK, result.StatusCode)
				select {
				case observed := <-received:
					assert.Equal(t, http.MethodPost, observed.method)
					assert.Equal(t, "/v1beta/openai/chat/completions", observed.path)
					assert.Equal(t, "Bearer gemini-fixture-key", observed.authorization)
					wire := unmarshalReasoningRequest(t, observed.body)
					assert.Equal(t, "gemini-4-"+model, wire["model"])
					assert.Equal(t, map[string]interface{}{"reasoning_effort": selected}, providerReasoningControls(wire))
					assert.NotContains(t, wire, "thinking_level")
					assert.NotContains(t, wire, "thinking_budget")
				default:
					t.Fatal("Gemini backend did not receive the request")
				}
			})
		}
	}
}

func TestGemini4RejectsUnsupportedReasoning(t *testing.T) {
	for _, model := range []string{"argon", "carbon", "hydrogen", "neon"} {
		t.Run(model+"/disabled", func(t *testing.T) {
			_, err := config.ParseYAMLBytes(gemini4Config(model, "http://127.0.0.1:8000", "use_reasoning: false"))
			require.ErrorContains(t, err, "always-on reasoning family")
		})
		unsupported := []string{"none", "minimal", "max"}
		if model == "argon" || model == "carbon" {
			unsupported = append(unsupported, "xhigh")
		}
		for _, effort := range unsupported {
			t.Run(model+"/"+effort, func(t *testing.T) {
				_, err := config.ParseYAMLBytes(gemini4Config(model, "http://127.0.0.1:8000",
					"use_reasoning: true\n            reasoning_effort: "+effort))
				require.ErrorContains(t, err, fmt.Sprintf("reasoning_effort %q is not supported", effort))
			})
		}
		for _, mode := range []string{config.ReasoningModeAdaptive, config.ReasoningModeDisabled} {
			t.Run(model+"/mode/"+mode, func(t *testing.T) {
				_, err := config.ParseYAMLBytes(gemini4Config(model, "http://127.0.0.1:8000",
					"use_reasoning: true\n            reasoning_mode: "+mode))
				require.ErrorContains(t, err, fmt.Sprintf("reasoning_mode %q is not supported", mode))
			})
		}
	}
}

func gemini4Config(model, endpoint, control string) []byte {
	return []byte(fmt.Sprintf(`version: v0.5
providers:
  models:
    - name: routed
      catalog: google/gemini-4-%s
      backend_refs:
        - name: primary
          provider: gemini
          endpoint: %s
          api_key: gemini-fixture-key
recipes:
  - name: default
    decisions:
      - name: gemini_default
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: routed
            %s
`, model, endpoint, control))
}
