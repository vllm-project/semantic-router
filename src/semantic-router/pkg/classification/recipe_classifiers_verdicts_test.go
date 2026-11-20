package classification

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

func TestRecipeClassifiersShareGuardTransportButNotNamespaces(t *testing.T) {
	_, store := sharedVerdictTestServer(t)
	document := strings.NewReplacer("TIMEOUT", "20000", "RULE\n", "").Replace(refusingGuardConfig)
	cfg, err := config.ParseYAMLBytes([]byte(document))
	require.NoError(t, err)
	cfg.GuardVerdicts = store
	other := cfg.Recipes[0]
	other.Name = "other"
	cfg.Recipes = append(cfg.Recipes, other, config.RoutingRecipe{Name: "plain"})
	for _, recipe := range cfg.Recipes {
		cfg.Entrypoints = append(cfg.Entrypoints, config.EntrypointMapping{ModelNames: []string{"vllm-sr/" + string(recipe.Name)}, Recipe: recipe.Name})
	}
	runtime, _ := servingtest.Runtime(t, map[string]runtimetest.Model{
		"primary": {Family: presets.FamilyVela3D3, MaxInputTokens: 32_768, Answer: findsInjection},
	})
	set, err := BuildRecipeClassifiers(cfg, RecipeRuntimeOptions{Runtime: runtime})
	require.NoError(t, err)
	t.Cleanup(func() { _ = set.Close() })
	require.NoError(t, set.InitializeRuntime())
	first, second := set.Default(), set.byRecipe["other"]
	require.NotNil(t, first.sharedVerdicts)
	require.Same(t, first.sharedVerdicts.sharedVerdictStore, second.sharedVerdicts.sharedVerdictStore)
	require.NotEqual(t, first.sharedVerdicts.key(windowVerdictKey{}), second.sharedVerdicts.key(windowVerdictKey{}))
	require.Nil(t, set.byRecipe["plain"].sharedVerdicts)
	require.False(t, first.ownsSharedVerdicts)
	require.NoError(t, first.Close())
	require.NoError(t, second.sharedVerdicts.client.Ping(context.Background()).Err(), "a recipe must not close the borrowed pool")
	require.NoError(t, set.Close())
	require.ErrorIs(t, set.sharedVerdicts.ctx.Err(), context.Canceled)
	require.Error(t, second.sharedVerdicts.client.Ping(context.Background()).Err())
	require.NoError(t, set.Close())
}

func TestRecipeWithoutGuardsDoesNotOpenVerdictStore(t *testing.T) {
	// Invalid runtime secrets are irrelevant when no request-stage guard uses them.
	cfg := &config.RouterConfig{GuardVerdicts: &config.GuardVerdictStoreConfig{
		Backend: "redis", HMACKeyEnv: "UNSET_GUARD_HMAC",
		Redis: &config.GuardVerdictRedisConfig{Address: "unreachable.example:6379", PasswordEnv: "UNSET_GUARD_PASSWORD"},
	}}
	set, err := BuildRecipeClassifiers(cfg)
	require.NoError(t, err)
	t.Cleanup(func() { _ = set.Close() })
	require.NoError(t, set.InitializeRuntime())
	require.Nil(t, set.sharedVerdicts)
	require.Nil(t, set.Default().sharedVerdicts)
}
