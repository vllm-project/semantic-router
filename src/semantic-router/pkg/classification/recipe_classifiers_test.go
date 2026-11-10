package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildRecipeClassifiersKeepsUnreachableRecipeValidationButSkipsRuntimeLifecycle(t *testing.T) {
	cfg := &config.RouterConfig{
		Recipes: []config.RoutingRecipe{
			{Name: config.DefaultRecipeName},
			{Name: "unmapped"},
			{Name: "mapped"},
		},
		Entrypoints: []config.EntrypointMapping{{
			ModelNames: []string{"vllm-sr/mapped"},
			Recipe:     "mapped",
		}},
	}

	classifiers, err := BuildRecipeClassifiers(cfg)
	if err != nil {
		t.Fatalf("BuildRecipeClassifiers() error = %v", err)
	}
	if _, ok := classifiers.ForRecipe("unmapped"); !ok {
		t.Fatal("unreachable recipe was not built and validated")
	}
	if len(classifiers.order) != 3 {
		t.Fatalf("built recipe order = %v, want all declared recipes", classifiers.order)
	}
	wantRuntime := []config.RecipeName{config.DefaultRecipeName, "mapped"}
	if len(classifiers.runtimeOrder) != len(wantRuntime) {
		t.Fatalf("runtime recipe order = %v, want %v", classifiers.runtimeOrder, wantRuntime)
	}
	for i := range wantRuntime {
		if classifiers.runtimeOrder[i] != wantRuntime[i] {
			t.Fatalf("runtime recipe order = %v, want %v", classifiers.runtimeOrder, wantRuntime)
		}
	}
	if len(classifiers.routingOrder) != len(wantRuntime) {
		t.Fatalf("routing recipe order = %v, want %v", classifiers.routingOrder, wantRuntime)
	}
}

func TestBuildRecipeClassifiersKeepsImplicitDefaultEntrypoint(t *testing.T) {
	cfg := &config.RouterConfig{
		Recipes: []config.RoutingRecipe{{Name: config.DefaultRecipeName}},
	}

	classifiers, err := BuildRecipeClassifiers(cfg)
	if err != nil {
		t.Fatalf("BuildRecipeClassifiers() error = %v", err)
	}
	if len(classifiers.runtimeOrder) != 1 ||
		classifiers.runtimeOrder[0] != config.DefaultRecipeName {
		t.Fatalf("default API lifecycle was lost: %v", classifiers.runtimeOrder)
	}
	if len(classifiers.routingOrder) != 1 || classifiers.routingOrder[0] != config.DefaultRecipeName {
		t.Fatalf("implicit default routing must remain reachable: %v", classifiers.routingOrder)
	}
}
