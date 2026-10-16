package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMemoryStartupRespectsDisabledDecisionPlugins(t *testing.T) {
	plugin := func(enabled bool) config.Decision {
		payload, err := config.NewStructuredPayload(config.MemoryPluginConfig{Enabled: enabled})
		if err != nil {
			t.Fatal(err)
		}
		return config.Decision{Name: "private", Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginMemory, Configuration: payload}}}
	}
	tests := []struct {
		name      string
		global    bool
		decisions []config.Decision
		recipes   []config.RoutingRecipe
		want      bool
	}{
		{name: "omitted"},
		{name: "disabled flat plugin", decisions: []config.Decision{plugin(false)}},
		{name: "enabled flat plugin", decisions: []config.Decision{plugin(true)}, want: true},
		{name: "global API opt in", global: true, decisions: []config.Decision{plugin(false)}, want: true},
		{name: "all recipe plugins disabled", recipes: []config.RoutingRecipe{
			{Name: "vault", Profile: config.RoutingProfile{Decisions: []config.Decision{plugin(false), plugin(false), plugin(false)}}},
			{Name: "balance", Profile: config.RoutingProfile{Decisions: []config.Decision{{Name: "simple"}}}},
		}},
		{name: "later recipe opts in", recipes: []config.RoutingRecipe{
			{Name: "vault", Profile: config.RoutingProfile{Decisions: []config.Decision{plugin(false)}}},
			{Name: "agent", Profile: config.RoutingProfile{Decisions: []config.Decision{plugin(true)}}},
		}, want: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &config.RouterConfig{Memory: config.MemoryConfig{Enabled: tc.global}, Recipes: tc.recipes}
			cfg.Decisions = tc.decisions
			if got := isMemoryEnabled(cfg); got != tc.want {
				t.Fatalf("backend initialization = %v, want %v", got, tc.want)
			}
		})
	}
}
