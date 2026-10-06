package config

import "testing"

func TestModelBindingProjectionKeepsCanonicalSourceImmutable(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.CategoryModel.ModelID = "models/default"
	cfg.ModelDeployments = map[string]ModelDeployment{"private": {Provider: ModelRuntimeProvider, Artifact: "/mounted/private"}}
	cfg.ModelBindings = map[string]ModelBinding{"domain_classifier": {Deployment: "private", Contract: "label_distribution.v1", Adapter: "mmbert32k", MappingPath: "/mounted/maps/domain.json"}}
	cfg.RoutingScope = "private"
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	projected, err := ProjectRecipeModelBindings(cfg, plan, "private")
	if err != nil {
		t.Fatal(err)
	}
	if projected.CategoryModel.ModelID != "/mounted/private" || projected.CategoryMappingPath != "/mounted/maps/domain.json" {
		t.Fatalf("incorrect projection: %#v", projected.CategoryModel)
	}
	if cfg.CategoryModel.ModelID != "models/default" || cfg.CategoryMappingPath != "" {
		t.Fatal("source mutated")
	}
	if _, err := ProjectRecipeModelBindings(cfg, plan, "other"); err == nil {
		t.Fatal("foreign recipe binding resolved")
	}
}

func TestModelBindingProjectionNamesTheModelAnAttachedRuntimeServes(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"domain-a": {Provider: ModelRuntimeProvider, Endpoint: "http://runtime:8100"},
		"guard-a":  {Provider: ModelRuntimeProvider, Endpoint: "unix:///run/runtime.sock", ServedName: "vela-guard"},
	}
	cfg.ModelBindings = map[string]ModelBinding{
		"domain_classifier": {Deployment: "domain-a", Contract: RemoteClassifierContractLabelDistribution, Adapter: "modernbert"},
		"prompt_guard":      {Deployment: "guard-a", Contract: RemoteClassifierContractLabelDistribution, Adapter: "modernbert"},
	}
	cfg.RoutingScope = "private"
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	projected, err := ProjectRecipeModelBindings(cfg, plan, "private")
	if err != nil {
		t.Fatal(err)
	}
	if projected.CategoryModel.ModelID != "domain-a" || !projected.IsCategoryClassifierEnabled() {
		t.Fatalf("an attached domain classifier without an artifact must stay enabled: %#v", projected.CategoryModel)
	}
	if projected.PromptGuard.ModelID != "vela-guard" {
		t.Fatalf("the guard must name the model its runtime serves: %q", projected.PromptGuard.ModelID)
	}
}

func TestServedModelPrefersTheArtifact(t *testing.T) {
	for _, tc := range []struct {
		deployment ModelDeployment
		want       string
	}{
		{ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain"}, "vllm-sr/Vela-1.0-Encoder-307M-Domain"},
		{ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "/models/domain", Endpoint: "http://runtime:8100"}, "/models/domain"},
		{ModelDeployment{Provider: ModelRuntimeProvider, Endpoint: "http://runtime:8100"}, "domain-a"},
		{ModelDeployment{Provider: ModelRuntimeProvider, Endpoint: "http://runtime:8100", ServedName: "vela-domain"}, "vela-domain"},
		{ModelDeployment{Provider: "http", Endpoint: "http://classifier:9000"}, ""},
	} {
		if got := tc.deployment.ServedModel("domain-a"); got != tc.want {
			t.Errorf("%+v: ServedModel = %q, want %q", tc.deployment, got, tc.want)
		}
	}
}
