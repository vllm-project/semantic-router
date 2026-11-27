package extproc

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

func museCatalogConfig(model, format, control string) []byte {
	return []byte(fmt.Sprintf(`version: v0.5
providers:
  models:
    - name: routed
      catalog: meta/%s
      api_format: %s
      backend_refs: [{name: primary, provider: meta, api_key: fixture-key}]
recipes:
  - name: default
    decisions:
      - name: route
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: routed, %s}]
`, model, format, control))
}

func TestMetaMuseCatalogRequest(t *testing.T) {
	for _, model := range []string{"muse-wonder-1.0", "muse-ember-1.0"} {
		for _, format := range []string{config.APIFormatChatCompletions, config.APIFormatResponses} {
			for _, level := range []string{"default", "minimal", "low", "medium", "high", "xhigh", "max"} {
				t.Run(model+"/"+format+"/"+level, func(t *testing.T) {
					checkMuseCatalogRequest(t, model, format, level)
				})
			}
		}
	}
}

func checkMuseCatalogRequest(t *testing.T, model, format, level string) {
	t.Helper()
	control := "use_reasoning: true"
	if level == "default" {
		level = "max"
	} else {
		control += ", reasoning_effort: " + level
	}
	cfg, err := config.ParseYAMLBytes(museCatalogConfig(model, format, control))
	require.NoError(t, err)
	decision := cfg.GetDecisionByName("route")
	require.NotNil(t, decision)
	router := &OpenAIRouter{
		Config:             cfg,
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
	}
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, nil)
	ctx.VSRSelectedDecision = decision
	request, immediate := router.prepareProtocolRequest([]byte(`{"model":"routed","messages":[{"role":"user","content":"Hello"}]}`), ctx)
	require.Nil(t, immediate)
	response, err := router.handleEntrypointModelRouting(request, "routed", decision.Name,
		entropy.ReasoningDecision{UseReasoning: true}, "routed", ctx)
	require.NoError(t, err)
	require.Nil(t, response.GetImmediateResponse())
	outbound := response.GetRequestBody().GetResponse()
	require.NotNil(t, outbound)
	headers := headerValuesByName(outbound.GetHeaderMutation().GetSetHeaders())
	body := unmarshalReasoningRequest(t, outbound.GetBodyMutation().GetBody())
	require.Equal(t, model, body["model"])
	require.Equal(t, "Bearer fixture-key", headers["Authorization"])
	wantPath := "/v1/chat/completions"
	wantControls := map[string]any{"reasoning_effort": level}
	if format == config.APIFormatResponses {
		wantPath = "/v1/responses"
		wantControls = map[string]any{"reasoning": map[string]any{"effort": level}}
	}
	require.Equal(t, wantPath, headers[":path"])
	require.Equal(t, wantControls, providerReasoningControls(body))
}

func TestMetaMuseCatalogRejectsUnsupportedReasoning(t *testing.T) {
	for _, model := range []string{"muse-wonder-1.0", "muse-ember-1.0"} {
		for _, test := range []struct{ control, wantErr string }{
			{"use_reasoning: false", "always-on reasoning family"},
			{"use_reasoning: true, reasoning_effort: none", `reasoning_effort "none" is not supported`},
			{"use_reasoning: true, reasoning_effort: standard", `reasoning_effort "standard" is not supported`},
		} {
			t.Run(model+"/"+test.control, func(t *testing.T) {
				_, err := config.ParseYAMLBytes(museCatalogConfig(model, config.APIFormatChatCompletions, test.control))
				require.ErrorContains(t, err, test.wantErr)
			})
		}
	}
}
