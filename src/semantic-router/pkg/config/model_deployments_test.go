package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func testDeploymentConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"shared-encoder": {Artifact: "models/maintained-encoder", Revision: strings.Repeat("f", 40), Provider: ModelRuntimeProvider, Device: "cpu", Input: ModelInputBudget{MaxTokens: 512, Overflow: "reject"}},
		"other-encoder":  {Artifact: "models/other-maintained-encoder", Provider: ModelRuntimeProvider, Device: "rocm:1"},
	}
	cfg.ModelAdmission = map[string]AdmissionConfig{"shared-encoder": {MaxConcurrency: 2, MaxQueue: 3, OnOverflow: "shed"}}
	cfg.Recipes = []RoutingRecipe{
		{Name: "support", Profile: RoutingProfile{ModelBindings: map[string]ModelBinding{
			"domain_classifier": {Deployment: "shared-encoder", Contract: RemoteClassifierContractLabelDistribution, Adapter: "modernbert", Head: "domain-head", MappingPath: "domain-labels.json"},
			"pii_classifier":    {Deployment: "shared-encoder", Contract: RemoteClassifierContractTokenSpans, Adapter: "modernbert", Head: "pii-head", MappingPath: "pii-labels.json"},
		}}},
		{Name: "coding", Profile: RoutingProfile{ModelBindings: map[string]ModelBinding{
			"domain_classifier": {Deployment: "other-encoder", Contract: RemoteClassifierContractLabelDistribution, Adapter: "mmbert32k"},
		}}},
	}
	return cfg
}

func TestCompileModelBindingsPreservesResourceAndTaskIdentity(t *testing.T) {
	cfg := testDeploymentConfig()
	plan, err := CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	domain, ok := plan.Lookup("support", "domain_classifier")
	if !ok || domain.Binding.Head != "domain-head" || domain.Admission.MaxConcurrency != 2 {
		t.Fatalf("domain=%+v", domain)
	}
	pii, ok := plan.Lookup("support", "pii_classifier")
	if !ok || pii.Deployment != domain.Deployment || pii.Binding.Head == domain.Binding.Head {
		t.Fatalf("head/resource identities conflated: %+v", pii)
	}
	if _, ok := plan.Lookup("coding", "pii_classifier"); ok {
		t.Fatal("foreign recipe lookup succeeded")
	}
	coding, _ := plan.Lookup("coding", "domain_classifier")
	if coding.Deployment.Device != "rocm:1" {
		t.Fatal("wrong recipe deployment")
	}
	cfg.ModelDeployments["shared-encoder"] = ModelDeployment{}
	retained, _ := plan.Lookup("support", "domain_classifier")
	if retained.Deployment.Artifact == "" {
		t.Fatal("compiled plan aliases mutable configuration")
	}
}

func TestCompileModelBindingsRejectsInvalidPreparation(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*RouterConfig)
		want   string
	}{
		{"unknown deployment", func(cfg *RouterConfig) { delete(cfg.ModelDeployments, "shared-encoder") }, "unknown deployment"},
		{"foreign provider", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Provider = "unknown"
			cfg.ModelDeployments["shared-encoder"] = d
		}, "unsupported provider"},
		{"malformed device", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Device = "CUDA 0"
			cfg.ModelDeployments["shared-encoder"] = d
		}, "device must be"},
		{"malformed profile", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Profile = "max-speed!"
			cfg.ModelDeployments["shared-encoder"] = d
		}, "profile must be"},
		{"negative budget", func(cfg *RouterConfig) {
			d := cfg.ModelDeployments["shared-encoder"]
			d.Input.MaxTokens = -1
			cfg.ModelDeployments["shared-encoder"] = d
		}, "must not be negative"},
		{"wrong contract", func(cfg *RouterConfig) {
			b := cfg.Recipes[0].Profile.ModelBindings["domain_classifier"]
			b.Contract = RemoteClassifierContractScore
			cfg.Recipes[0].Profile.ModelBindings["domain_classifier"] = b
		}, "contract must"},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := testDeploymentConfig()
			test.mutate(cfg)
			_, err := CompileModelBindings(cfg)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got %v, want %q", err, test.want)
			}
		})
	}
}

func TestPluginAcceleratorAndProfileNamesAreTheRuntimes(t *testing.T) {
	cfg := testDeploymentConfig()
	d := cfg.ModelDeployments["shared-encoder"]
	d.Device, d.Profile = "npu:1", "vendor_low_latency"
	cfg.ModelDeployments["shared-encoder"] = d
	if _, err := CompileModelBindings(cfg); err != nil {
		t.Fatalf("a well-formed plugin accelerator and profile compile; the runtime decides whether it has them: %v", err)
	}
}

func TestNamedDeploymentAdmissionAndCanonicalRoundTrip(t *testing.T) {
	cfg := testDeploymentConfig()
	deployment := cfg.ModelDeployments["other-encoder"]
	deployment.Input.MaxTokens = 8192
	cfg.ModelDeployments["other-encoder"] = deployment
	if err := validateModelAdmissionContracts(cfg); err != nil {
		t.Fatal(err)
	}
	global := canonicalModelCatalogFromRouterConfig(cfg)
	data, err := yaml.Marshal(global)
	if err != nil {
		t.Fatal(err)
	}
	var decoded CanonicalModelCatalog
	if err := yaml.UnmarshalStrict(data, &decoded); err != nil {
		t.Fatal(err)
	}
	var roundTrip RouterConfig
	applyCanonicalModelCatalogGlobal(&roundTrip, decoded)
	if roundTrip.ModelDeployments["other-encoder"] != cfg.ModelDeployments["other-encoder"] {
		t.Fatal("deployment lost during canonical round trip")
	}
	recipes := canonicalRecipesFromRouterConfig(cfg)
	if len(recipes) != 2 || recipes[0].Routing.ModelBindings["pii_classifier"].Head != "pii-head" {
		t.Fatal("recipe binding lost during export")
	}
	scoped := cfg.ConfigForRecipe(&cfg.Recipes[0])
	if scoped.ModelBindings["domain_classifier"].Deployment != "shared-encoder" {
		t.Fatal("recipe view lost bindings")
	}
}

func TestRemovedClassifierSessionBankOptionIsRejected(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
global:
  model_catalog:
    deployments:
      classifier:
        artifact: /models/classifier
        provider: model_runtime
        device: rocm:0
        input:
          max_tokens: 8192
          overflow: reject
        short_sequence_tokens: 512
`))
	if err == nil || !strings.Contains(err.Error(), `unknown field "short_sequence_tokens"`) {
		t.Fatalf("removed session-bank option must be rejected, got %v", err)
	}
}

func TestDormantComplexityRejectsUnsupportedLocalProvider(t *testing.T) {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{"local": {Provider: ModelRuntimeProvider, Artifact: "/mounted/local"}}
	cfg.Recipes = []RoutingRecipe{{Name: "dormant", Profile: RoutingProfile{ModelBindings: map[string]ModelBinding{"complexity": {Deployment: "local", Contract: "score.v1", Adapter: "mmbert"}}}}}
	if _, err := CompileModelBindings(cfg); err == nil {
		t.Fatal("dormant recipe accepted unsupported local complexity")
	}
}
