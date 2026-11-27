package extproc

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestOpenAICatalogEffortBoundaries(t *testing.T) {
	for _, model := range []string{"gpt-6.2-astra", "gpt-6.2-sol", "gpt-6.1-terra"} {
		for _, format := range []string{config.APIFormatChatCompletions, config.APIFormatResponses} {
			for _, effort := range []string{"none", "minimal", "low", "max"} {
				t.Run(model+"/"+format+"/"+effort, func(t *testing.T) {
					checkOpenAICatalogEffort(t, model, format, effort)
				})
			}
		}
	}
}

func checkOpenAICatalogEffort(t *testing.T, model, format, effort string) {
	t.Helper()
	wire := catalogReasoningWireCase{catalog: "openai/" + model, provider: "openai", apiFormat: format}
	control := "use_reasoning: false"
	if effort == "none" {
		wire.mode = config.ReasoningModeDisabled
	} else {
		wire.enabled, wire.effort = true, effort
		control = "use_reasoning: true, reasoning_effort: " + effort
	}
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`
version: v0.5
providers:
  models:
    - name: routed
      catalog: openai/%s
      api_format: %s
      backend_refs: [{name: primary, provider: openai}]
recipes:
  - name: default
    decisions:
      - name: route
        priority: 1
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: routed, %s}]
`, model, format, control)))
	if effort == "none" && model != "gpt-6.1-terra" {
		require.ErrorContains(t, err, "always-on reasoning family")
		return
	}
	if effort == "minimal" {
		require.ErrorContains(t, err, `reasoning_effort "minimal" is not supported`)
		return
	}
	require.NoError(t, err)
	require.Equal(t, model, cfg.ResolveExternalModelID("routed", "routed_primary"))
	request, _ := renderCatalogReasoningWire(t, wire)
	want := map[string]any{"reasoning_effort": effort}
	if format == config.APIFormatResponses {
		want = map[string]any{"reasoning": map[string]any{"effort": effort}}
	}
	require.Equal(t, want, providerReasoningControls(request))
}
