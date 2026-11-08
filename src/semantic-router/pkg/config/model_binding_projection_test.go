package config

import "testing"

func TestABindingResolvesByNameAndLeavesTheSourceUntouched(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.CategoryModel.ModelID = "models/default"
	cfg.ModelDeployments = map[string]ModelDeployment{"private": {Provider: ModelRuntimeProvider, Artifact: "/mounted/private"}}
	cfg.ModelBindings = map[string]ModelBinding{"domain_classifier": {Deployment: "private", Contract: "label_distribution.v1", Adapter: "mmbert32k", MappingPath: "/mounted/maps/domain.json"}}
	cfg.RoutingScope = "private"
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	spec, ok := plan.Lookup("private", "domain_classifier")
	if !ok || spec.Binding.Deployment != "private" || spec.Deployment.Artifact != "/mounted/private" || spec.Binding.MappingPath != "/mounted/maps/domain.json" {
		t.Fatalf("binding resolved to %+v, %v", spec, ok)
	}
	if cfg.CategoryModel.ModelID != "models/default" || cfg.CategoryMappingPath != "" {
		t.Fatal("resolution wrote the binding back into the module")
	}
	if _, ok := plan.Lookup("other", "domain_classifier"); ok {
		t.Fatal("foreign recipe binding resolved")
	}
}

func TestAnAttachedRuntimeAnswersWithTheModelItServes(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"domain-a": {Provider: ModelRuntimeProvider, Endpoint: "http://runtime:8100"},
		"guard-a":  {Provider: ModelRuntimeProvider, Endpoint: "unix:///run/runtime.sock", ServedName: "vela-guard"},
		"halu-a":   {Provider: ModelRuntimeProvider, Endpoint: "https://runtime.example.com:8100", Model: "vela-halu"},
	}
	cfg.ModelBindings = map[string]ModelBinding{
		"domain":        {Deployment: "domain-a"},
		"jailbreak":     {Deployment: "guard-a"},
		"hallucination": {Deployment: "halu-a"},
	}
	cfg.RoutingScope = "private"
	for consumer, want := range map[string]string{"domain_classifier": "domain-a", "prompt_guard": "vela-guard", "hallucination_detector": "vela-halu"} {
		resolved, ok, err := cfg.ResolveTask(consumer)
		if err != nil || !ok {
			t.Fatalf("%s: %v %v", consumer, ok, err)
		}
		if got := resolved.Deployment.ServedModel(resolved.Name); got != want {
			t.Fatalf("%s answers with %q, want %q", consumer, got, want)
		}
	}
	if !cfg.IsCategoryClassifierEnabled() {
		t.Fatal("an attached domain classifier without an artifact must stay enabled")
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
