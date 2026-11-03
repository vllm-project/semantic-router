package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A decision's allow and block lists authorize tools; advanced filtering only
// tunes retrieval. These tests turn advanced filtering off explicitly and
// check that the decision's block list still holds for retrieval, for the
// sticky eligible catalog, and for the session state sticky selection seeds.

// stickyMixedQuery is relevant to weather first and calendar second, so a
// top-1 retrieval returns weather unless a block rule removes it.
const stickyMixedQuery = "weather for the team meeting"

var decisionPolicyAdvancedFiltering = map[string]*config.AdvancedToolFilteringConfig{
	"advanced_filtering_unset":    nil,
	"advanced_filtering_disabled": {Enabled: false},
}

func blockWeatherToolsConfig(base *config.ToolsPluginConfig) *config.ToolsPluginConfig {
	base.Mode = config.ToolsPluginModeFiltered
	base.BlockTools = []string{"weather"}
	return base
}

func TestStickyDecisionBlockListHoldsWithoutAdvancedFiltering(t *testing.T) {
	for name, advanced := range decisionPolicyAdvancedFiltering {
		for _, format := range stickyTestFormats {
			t.Run(name+"/"+string(format), func(t *testing.T) {
				decision := stickyDecision(t, "assistant-tools", &config.ToolSelectionPluginConfig{
					Enabled: true, Mode: config.ToolSelectionModeAdd, TopK: 1,
					Sticky: &config.StickyToolSelectionConfig{Enabled: true, MaxTools: intPtr(3)},
				}, blockWeatherToolsConfig(supportedStickyToolsConfig()))
				h := newStickyHarness(t, format, decision, nil)
				h.router.Config.Tools.AdvancedFiltering = advanced
				h.provider.vectors[stickyMixedQuery] = []float32{0.8, 0.6, 0, 0}

				first := h.run(stickyTurn{query: stickyMixedQuery})
				require.Equal(t, []string{"calendar"}, first.tools, "a blocked tool must not be selected or seeded")
				require.Equal(t, "seeded", first.receipt.Outcome)

				second := h.run(stickyTurn{query: stickyTestQueries["calendar"], called: []string{"weather"}})
				require.Equal(t, []string{"calendar"}, second.tools, "history that called a blocked tool must not pin it")
				require.Equal(t, "reused", second.receipt.Outcome)
				require.Zero(t, second.receipt.Pinned)
			})
		}
	}
}

// The stateless paths share retrieval with sticky selection, so the same
// configuration must not emit the blocked tool through tool_selection add
// mode or the tools plugin's own semantic selection.
func TestStatelessDecisionBlockListHoldsWithoutAdvancedFiltering(t *testing.T) {
	blocked := blockWeatherToolsConfig(&config.ToolsPluginConfig{Enabled: true})
	decisions := map[string]config.Decision{
		"tool_selection_add": stickyDecision(t, "assistant-tools", &config.ToolSelectionPluginConfig{
			Enabled: true, Mode: config.ToolSelectionModeAdd, TopK: 1,
		}, blocked),
		"tools_plugin_selection": {Name: "assistant-tools", Plugins: []config.DecisionPlugin{mustToolsDecisionPlugin(t, blocked)}},
	}
	for name, advanced := range decisionPolicyAdvancedFiltering {
		for path, decision := range decisions {
			t.Run(name+"/"+path, func(t *testing.T) {
				h := newStickyHarness(t, llmprotocol.OpenAIChatV1, supportedStickyDecision(t, "assistant-tools", config.ToolSelectionModeAdd), nil)
				h.decision = decision
				h.router.Config.Tools.AdvancedFiltering = advanced
				h.provider.vectors[stickyMixedQuery] = []float32{0.8, 0.6, 0, 0}

				result := h.run(stickyTurn{query: stickyMixedQuery})
				require.Equal(t, []string{"calendar"}, result.tools)
				require.Nil(t, result.receipt, "a non-sticky decision records no sticky receipt")
				require.Zero(t, h.store.operations())
			})
		}
	}
}
