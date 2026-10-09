package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func reflectionDecision(name, algorithm string) config.Decision {
	return config.Decision{
		Name: name,
		Plugins: []config.DecisionPlugin{{
			Type: "memory",
			Configuration: config.MustStructuredPayload(map[string]interface{}{
				"enabled":    false,
				"reflection": map[string]interface{}{"algorithm": algorithm},
			}),
		}},
	}
}

func TestCreateMemoryRuntimeValidatesReflectionAlgorithm(t *testing.T) {
	scopes := []struct {
		name  string
		field string
		build func(algorithm string) *config.RouterConfig
	}{
		{
			name:  "global",
			field: "global.stores.memory.reflection.algorithm",
			build: func(algorithm string) *config.RouterConfig {
				cfg := &config.RouterConfig{}
				cfg.Memory.Reflection.Algorithm = algorithm
				return cfg
			},
		},
		{
			name:  "flat_decision",
			field: "routing.decisions[memory_route].plugins[memory].reflection.algorithm",
			build: func(algorithm string) *config.RouterConfig {
				cfg := &config.RouterConfig{}
				cfg.Decisions = []config.Decision{reflectionDecision("memory_route", algorithm)}
				return cfg
			},
		},
		{
			name:  "recipe_decision",
			field: "routing.decisions[recipe_memory_route].plugins[memory].reflection.algorithm",
			build: func(algorithm string) *config.RouterConfig {
				cfg := &config.RouterConfig{}
				cfg.Recipes = []config.RoutingRecipe{
					{Name: config.DefaultRecipeName},
					{Name: "private", Profile: config.RoutingProfile{Decisions: []config.Decision{
						reflectionDecision("recipe_memory_route", algorithm),
					}}},
				}
				return cfg
			},
		},
	}
	algorithms := []struct {
		algorithm string
		wantErr   bool
		hint      string
	}{
		{algorithm: "", wantErr: false},
		{algorithm: "heuristic", wantErr: false},
		{algorithm: "noop", wantErr: false},
		{algorithm: "recency_semantic", wantErr: true, hint: `use "heuristic"`},
		{algorithm: "Heuristic", wantErr: true},
		{algorithm: "mmr", wantErr: true},
	}

	for _, scope := range scopes {
		for _, tc := range algorithms {
			t.Run(scope.name+"/"+tc.algorithm, func(t *testing.T) {
				// Memory stays disabled, so no store is created and only the
				// startup check can fail.
				store, extractor, err := createMemoryRuntime(scope.build(tc.algorithm))
				assert.Nil(t, store)
				assert.Nil(t, extractor)
				if !tc.wantErr {
					require.NoError(t, err)
					return
				}
				require.Error(t, err)
				for _, want := range []string{scope.field, `"` + tc.algorithm + `"`, "registered: heuristic, noop", tc.hint} {
					assert.Contains(t, err.Error(), want)
				}
			})
		}
	}
}

// A valid decision override does not hide an unknown global value, because
// decisions without an override still inherit it.
func TestCreateMemoryRuntimeRejectsUnknownGlobalAlgorithmDespiteOverride(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Memory.Reflection.Algorithm = "recency_semantic"
	cfg.Decisions = []config.Decision{reflectionDecision("memory_route", "heuristic")}

	_, _, err := createMemoryRuntime(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "global.stores.memory.reflection.algorithm")
}
