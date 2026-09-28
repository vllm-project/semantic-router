package modelruntime

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestGlobalCacheNLIUsesDemandAndCanonicalSource(t *testing.T) {
	for _, explicit := range []bool{false, true} {
		t.Run(map[bool]string{false: "module_default", true: "global_binding"}[explicit], func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.SemanticCache.Enabled = true
			cfg.SemanticCache.PolarityGuard = &config.PolarityGuardConfig{Mode: "nli"}
			cfg.HallucinationMitigation.NLIModel.ModelID = nliServiceFixture(t, 2)
			cfg.HallucinationMitigation.NLIModel.UseCPU = true
			cfg.ModelDeployments = map[string]config.ModelDeployment{
				"global": {Provider: "candle", Device: "cpu", Precision: "native", Artifact: cfg.HallucinationMitigation.NLIModel.ModelID},
				"first":  {Provider: "candle", Device: "cpu", Precision: "native", Artifact: nliServiceFixture(t, 0)},
				"second": {Provider: "candle", Device: "cpu", Precision: "native", Artifact: nliServiceFixture(t, 1)},
			}
			decl := func(name string) config.ModelBinding {
				return config.ModelBinding{Deployment: name, Adapter: "modernbert", Contract: "text_pair_distribution.v1"}
			}
			cfg.ModelBindings = map[string]config.ModelBinding{"hallucination_explainer": decl("first")}
			cfg.Recipes = []config.RoutingRecipe{
				{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{ModelBindings: cfg.ModelBindings}},
				{Name: "second", Profile: config.RoutingProfile{ModelBindings: map[string]config.ModelBinding{"hallucination_explainer": decl("second")}}},
			}
			if explicit {
				cfg.GlobalModelBindings = map[string]config.ModelBinding{"hallucination_explainer": decl("global")}
				cfg.HallucinationMitigation.NLIModel.ModelID = "/unused/implicit-model"
			}
			runtime := native.New(binding.NewPool())
			service, err := PrepareOwnedResponseCacheNLI(context.Background(), cfg, runtime)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = service.Close() })
			input := tasks.TextPairRequest{Premise: "hello", Hypothesis: "world"}
			got, err := service.Call(context.Background(), string(config.GlobalModelScope), input)
			if err != nil || got.Probabilities[2] < 0.8 {
				t.Fatalf("global cache borrowed recipe NLI: %+v / %v", got, err)
			}
			plan, err := config.CompileModelBindings(cfg)
			if err != nil {
				t.Fatal(err)
			}
			for _, scope := range []config.RecipeName{config.DefaultRecipeName, "second"} {
				spec, _ := plan.Lookup(scope, "hallucination_explainer")
				handle, prepareErr := runtime.TextPair(context.Background(), spec)
				if prepareErr != nil {
					t.Fatal(prepareErr)
				}
				t.Cleanup(func() { _ = handle.Close() })
				result, callErr := handle.Call(context.Background(), string(scope), input)
				winner := 0
				if scope == "second" {
					winner = 1
				}
				if callErr != nil || result.Probabilities[winner] < 0.8 {
					t.Fatalf("recipe override lost: %+v / %v", result, callErr)
				}
			}
			candidate := native.New(runtime.Pool)
			replacement, err := PrepareOwnedResponseCacheNLI(context.Background(), cfg, candidate)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = replacement.Close() })
			oldID := ""
			for _, entry := range runtime.PreparedBindings() {
				if entry.Identity.Recipe == string(config.GlobalModelScope) {
					oldID = entry.ResourceID
				}
			}
			if got := candidate.PreparedBindings(); len(got) != 1 || got[0].ResourceID != oldID || oldID == "" {
				t.Fatalf("reload lost shared resource: %+v", got)
			}
			if err := service.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err := replacement.Call(context.Background(), string(config.GlobalModelScope), input); err != nil {
				t.Fatal("retiring old cache unloaded replacement", err)
			}
			if _, err := replacement.Call(context.Background(), "second", input); !errors.Is(err, binding.ErrCapability) {
				t.Fatalf("service handle leaked recipe scope: %v", err)
			}
		})
	}
}

func TestGlobalCacheNLIDoesNotLoadWithoutSemanticConsumer(t *testing.T) {
	for _, mode := range []string{"disabled", "unused", "exact", "plugin_disabled"} {
		t.Run(mode, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.SemanticCache.Enabled = mode != "disabled"
			cfg.SemanticCache.PolarityGuard = &config.PolarityGuardConfig{Mode: "nli"}
			cfg.HallucinationMitigation.NLIModel.ModelID = "/missing/unused-nli"
			cfg.Decisions = []config.Decision{{Name: "route"}}
			if mode == "exact" || mode == "plugin_disabled" {
				cfg.Decisions[0].Plugins = []config.DecisionPlugin{{Type: "response_cache", Configuration: config.MustStructuredPayload(config.ResponseCachePluginConfig{Enabled: mode == "exact", Mode: "exact"})}}
			}
			runtime := native.New(nil)
			handle, err := PrepareOwnedResponseCacheNLI(context.Background(), cfg, runtime)
			if err != nil || handle != nil || len(runtime.PreparedBindings()) != 0 {
				t.Fatalf("unused NLI loaded: %v / %v", handle, err)
			}
		})
	}
	cfg := &config.RouterConfig{}
	cfg.SemanticCache.Enabled = true
	cfg.SemanticCache.PolarityGuard = &config.PolarityGuardConfig{Mode: "nli"}
	cfg.HallucinationMitigation.NLIModel.ModelID = "/missing/demanded-nli"
	if _, err := PrepareOwnedResponseCacheNLI(context.Background(), cfg, native.New(nil)); err == nil {
		t.Fatal("demanded invalid NLI did not fail startup")
	}
}
