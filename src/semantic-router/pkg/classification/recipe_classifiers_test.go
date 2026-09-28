package classification

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestUnreachableDefaultDoesNotPrepareRoutingArtifacts(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "unprovisioned")
	defaultProfile := config.RoutingProfile{
		Signals:       config.Signals{ClassifierRules: []config.ClassifierSignalRule{{Name: "risk", Type: config.ClassifierSignalTypeLocal, ModelPath: missing, UseCPU: true, Labels: []string{"safe", "unsafe"}}}},
		ModelBindings: map[string]config.ModelBinding{"classifier.risk": {Deployment: "dormant", Adapter: "auto", Contract: config.RemoteClassifierContractLabelDistribution}},
	}
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{}},
		Recipes:       []config.RoutingRecipe{{Name: config.DefaultRecipeName, Profile: defaultProfile}, {Name: "active"}},
		Entrypoints:   []config.EntrypointMapping{{ModelNames: []string{"public"}, Recipe: "active"}},
	}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"dormant": {Artifact: missing, Provider: "candle", Device: "cpu"}}
	classifiers, err := BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("unreachable default tried to load routing model: %v", err)
	}
	t.Cleanup(func() { _ = classifiers.Close() })
	if len(classifiers.Default().genericClassifiers) != 0 {
		t.Fatal("unreachable routing model was prepared")
	}
	if initErr := classifiers.InitializeRuntime(); initErr != nil {
		t.Fatal(initErr)
	}
	if len(cfg.Recipes[0].Profile.Signals.ClassifierRules) != 1 {
		t.Fatal("preparation mutated the canonical recipe")
	}
	invalid := *cfg
	invalid.ModelBindings = map[string]config.ModelBinding{"classifier.missing": {Deployment: "dormant", Adapter: "auto", Contract: config.RemoteClassifierContractLabelDistribution}}
	invalid.Recipes = nil
	if _, invalidErr := BuildRecipeClassifiers(&invalid, nil, nil, nil); invalidErr == nil {
		t.Fatal("unreachable default bypassed canonical binding validation")
	}
}

func TestRecipeClassifierReadinessSeparatesDefaultAPIFromInventory(t *testing.T) {
	defaultClassifier := &Classifier{}
	namedClassifier := &Classifier{
		factCheckClassifier: &FactCheckClassifier{initialized: true},
		hallucinationDetector: &HallucinationDetector{
			initialized:    true,
			nliInitialized: true,
		},
		feedbackDetector: &FeedbackDetector{initialized: true},
	}
	classifiers := &RecipeClassifiers{
		byRecipe: map[config.RecipeName]*Classifier{
			config.DefaultRecipeName: defaultClassifier,
			"model-backed":           namedClassifier,
		},
		order: []config.RecipeName{config.DefaultRecipeName, "model-backed"},
	}

	if classifiers.HasFactCheckClassifier() ||
		classifiers.HasHallucinationDetector() ||
		classifiers.HasHallucinationExplainer() ||
		classifiers.HasFeedbackDetector() {
		t.Fatal("named-only models must not make default model-less APIs ready")
	}
	if !classifiers.HasAnyFactCheckClassifier() {
		t.Fatal("named-recipe fact-check readiness was not included in inventory")
	}
	if !classifiers.HasAnyHallucinationDetector() {
		t.Fatal("named-recipe hallucination readiness was not aggregated")
	}
	if !classifiers.HasAnyHallucinationExplainer() {
		t.Fatal("named-recipe hallucination explainer readiness was not aggregated")
	}
	if !classifiers.HasAnyFeedbackDetector() {
		t.Fatal("named-recipe feedback readiness was not aggregated")
	}
}

func TestRecipeClassifierReadinessStaysFalseWhenModelsAreUninitialized(t *testing.T) {
	classifiers := &RecipeClassifiers{
		byRecipe: map[config.RecipeName]*Classifier{
			config.DefaultRecipeName: {},
			"model-backed": {
				factCheckClassifier:   &FactCheckClassifier{},
				hallucinationDetector: &HallucinationDetector{},
				feedbackDetector:      &FeedbackDetector{},
			},
		},
		order: []config.RecipeName{config.DefaultRecipeName, "model-backed"},
	}

	if classifiers.HasAnyFactCheckClassifier() ||
		classifiers.HasAnyHallucinationDetector() ||
		classifiers.HasAnyHallucinationExplainer() ||
		classifiers.HasAnyFeedbackDetector() {
		t.Fatal("uninitialized named-recipe models must not report ready")
	}
}

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

	classifiers, err := BuildRecipeClassifiers(cfg, nil, nil, nil)
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

func TestBuildRecipeClassifiersKeepsDefaultAPIWhenAutoRoutingIsDisabled(t *testing.T) {
	cfg := &config.RouterConfig{
		RouterOptions: config.RouterOptions{AutoModelNames: []string{}},
		Recipes:       []config.RoutingRecipe{{Name: config.DefaultRecipeName}},
	}

	classifiers, err := BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildRecipeClassifiers() error = %v", err)
	}
	if len(classifiers.runtimeOrder) != 1 ||
		classifiers.runtimeOrder[0] != config.DefaultRecipeName {
		t.Fatalf("default API lifecycle was lost: %v", classifiers.runtimeOrder)
	}
	if len(classifiers.routingOrder) != 0 {
		t.Fatalf("disabled default routing remained reachable: %v", classifiers.routingOrder)
	}
}
