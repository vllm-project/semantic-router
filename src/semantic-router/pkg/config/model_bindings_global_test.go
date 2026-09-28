package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestGlobalModelBindingsKeepServiceAndRecipeOverridesIsolated(t *testing.T) {
	cfg := testDeploymentConfig()
	global := ModelBinding{Deployment: "shared-encoder", Contract: "embedding.v1", Adapter: "mmbert", Head: "onnx/model.onnx"}
	cfg.GlobalModelBindings = map[string]ModelBinding{"embedding": global}
	local := global
	local.Deployment = "other-encoder"
	cfg.Recipes[1].Profile.ModelBindings["embedding"] = local
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	service, ok := plan.LookupGlobal("embedding")
	if !ok || service.Recipe != GlobalModelScope || service.Binding != global {
		t.Fatalf("service = %+v", service)
	}
	for _, name := range []RecipeName{"support", "coding"} {
		spec, ok := plan.Lookup(name, "embedding")
		if !ok || spec.Recipe != name {
			t.Fatalf("scope %s = %+v", name, spec)
		}
		want := global
		if name == "coding" {
			want = local
		}
		if spec.Binding != want {
			t.Fatalf("scope %s borrowed another binding: %+v", name, spec)
		}
	}
	if _, ok := plan.Lookup("foreign", "embedding"); ok {
		t.Fatal("global catalog allowed foreign recipe lookup")
	}
	if _, ok := cfg.Recipes[0].Profile.ModelBindings["embedding"]; ok {
		t.Fatal("compilation mutated authoring config")
	}
	projected, err := ProjectRecipeModelBindings(cfg.ConfigForRecipe(&cfg.Recipes[0]), plan, "support")
	if err != nil || projected.ModelBindings["embedding"] != global {
		t.Fatalf("projection = %+v, %v", projected, err)
	}
	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	var document CanonicalConfig
	if err := yaml.UnmarshalStrict(encoded, &document); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(document.Global.ModelCatalog.Bindings, cfg.GlobalModelBindings) {
		t.Fatal("global catalog was lost in export")
	}
	for _, recipe := range document.Recipes {
		if recipe.Name == "support" && recipe.Routing.ModelBindings["embedding"].Deployment != "" {
			t.Fatal("export expanded inherited bindings")
		}
	}
}

func TestGlobalRuleBindingsAreVisibleOnlyToDeclaringRecipes(t *testing.T) {
	cfg := testDeploymentConfig()
	cfg.GlobalModelBindings = map[string]ModelBinding{
		"classifier.risk": {Deployment: "shared-encoder", Contract: "label_distribution.v1", Adapter: "modernbert"},
		"safety.policy":   {Deployment: "shared-encoder", Contract: "label_distribution.v1", Adapter: "modernbert"},
	}
	if got := cfg.EffectiveModelBindings(Signals{}, nil); len(got) != 0 {
		t.Fatalf("foreign symbols inherited: %+v", got)
	}
	rules := Signals{ClassifierRules: []ClassifierSignalRule{{Name: "risk"}}, SafetyRules: []SafetyRule{{Name: "policy"}}}
	if got := cfg.EffectiveModelBindings(rules, nil); len(got) != 2 {
		t.Fatalf("declared rules lost global serving: %+v", got)
	}
	local := map[string]ModelBinding{"classifier.missing": cfg.GlobalModelBindings["classifier.risk"]}
	if _, ok := cfg.EffectiveModelBindings(Signals{}, local)["classifier.missing"]; !ok {
		t.Fatal("invalid explicit binding was silently discarded instead of validated")
	}
}

func TestGlobalModelBindingDeclarationsValidateBeforeProvisioning(t *testing.T) {
	cfg := testDeploymentConfig()
	cfg.GlobalModelBindings = map[string]ModelBinding{"embedding": {Deployment: "missing", Contract: "embedding.v1", Adapter: "mmbert"}}
	if _, err := CompileModelBindings(cfg); err == nil || !strings.Contains(err.Error(), "global.model_catalog.bindings.embedding") {
		t.Fatalf("global declaration error = %v", err)
	}
}
